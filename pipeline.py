"""
Includes ASR (OpenAI Whisper-Tiny), Text Summarization (BART), and Extractive QA (RoBERTa + BERT-tiny).
All models are lazily loaded.
"""

import os
import re
import warnings

import torch
import spacy
from transformers import pipeline, AutoTokenizer

# Suppress non-critical logs
warnings.filterwarnings("ignore")

# GPU/CPU Device Configuration
device = 0 if torch.cuda.is_available() else -1

# Global model caches
_nlp = None
_qa_pipelines = None
_summary_pipeline = None
_speech_pipeline = None

# Model relative weights for ensemble voting
MODEL_WEIGHTS = {
    "roberta": 1.0,
    "bert-tiny": 0.25,
}


# Audio Preprocessing
def load_audio_waveform(audio_source, target_sr=16000):
    """
    Load an audio/video file and convert it into a waveform
    suitable for Whisper (mono, 16 kHz, float32).
    """

    # Import pydub for audio decoding
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError(
            "pydub is required for audio decoding. Install it with "
            "'pip install pydub' (and ensure ffmpeg is installed and on PATH)."
        ) from e

    # Read the audio (ffmpeg handles extracting audio from video files)
    try:
        audio = AudioSegment.from_file(audio_source)
    except Exception as e:
        raise RuntimeError(
            "Could not decode the audio. Ensure ffmpeg is installed "
            f"and the file contains a valid audio track.\nOriginal error: {e}"
        )

    # Converting audio to Whisper's required format:
    # 16 kHz sampling rate
    # Mono (single channel)
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    import numpy as np

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize integer samples to the range [-1, 1]
    max_val = float(1 << (8 * audio.sample_width - 1))
    samples /= max_val

    return samples, target_sr


def load_speech_pipeline():

    # Access the global variable to avoid loading the model multiple times
    global _speech_pipeline
    if _speech_pipeline is None:
        try:
            _speech_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device,
                chunk_length_s=30,  # Process audio in 30-second chunks
                stride_length_s=5,  # 5-second overlap between chunks
            )
        except Exception as e:
            raise RuntimeError(f"Could not load Whisper model: {e}")
    return _speech_pipeline


# convert uploaded audio/video file into a transcript
def transcribe_audio(audio_source, filename_hint=None):

    if audio_source is None:
        raise ValueError("No audio source provided.")

    # If the input is raw bytes, convert it into a file-like object
    if isinstance(audio_source, (bytes, bytearray)):
        import io

        buf = io.BytesIO(audio_source)
        if filename_hint:
            buf.name = filename_hint
        audio_source = buf

    # If a file path is given, verify that it exists
    elif isinstance(audio_source, str):
        if not os.path.exists(audio_source):
            raise ValueError(f"Audio file path does not exist: {audio_source}")

    # Load the Whisper ASR pipeline
    asr = load_speech_pipeline()
    try:
        waveform, sampling_rate = load_audio_waveform(
            audio_source
        )  # Convert audio/video into a normalized waveform (NumPy array)

        # Pass the waveform to Whisper for speech-to-text transcription
        result = asr(
            {"array": waveform, "sampling_rate": sampling_rate}, return_timestamps=True
        )
        return result.get("text", "").strip()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


# SpaCy Used for
# 1. Sentence Splitting
# 2. POS (Part-of-Speech) Tagging


# Load the spaCy English model
def get_spacy_nlp():  # helper function

    global _nlp
    # Load the model only once (Lazy Loading)
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys

            warnings.warn(
                "spaCy model 'en_core_web_sm' not found. Downloading it now..."
            )
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_qa_pipelines():

    global _qa_pipelines
    if _qa_pipelines is None:
        _qa_pipelines = []
        QA_MODELS = [
            ("deepset/roberta-base-squad2", "roberta"),
            ("mrm8488/bert-tiny-finetuned-squadv2", "bert-tiny"),
        ]
        for model_name, model_tag in QA_MODELS:
            try:
                qa = pipeline(
                    "question-answering",  # Create a QA pipeline
                    model=model_name,  # Load the pretrained model
                    tokenizer=AutoTokenizer.from_pretrained(model_name),
                    device=device,
                )
                _qa_pipelines.append(
                    (qa, model_tag)
                )  # Storing the pipeline along with its model name
            except Exception as e:
                warnings.warn(f"Could not load QA model '{model_name}': {e}")

        if not _qa_pipelines:
            raise RuntimeError("No QA pipelines could be loaded.")
    return _qa_pipelines


def load_summary_pipeline():

    global _summary_pipeline
    if _summary_pipeline is None:
        try:
            # Create a Hugging Face summarization pipeline using BART
            _summary_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",  # BART summarization model
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load summarization model 'facebook/bart-large-cnn': {e}"
            ) from e
    return _summary_pipeline


def clean_answer(answer):
    # Normalize whitespace and remove subword tokenization artifacts

    answer = re.sub(  # Replace multiple spaces/newlines with a single space
        r"\s+", " ", answer
    ).strip()
    answer = re.sub(  # Remove subword tokenization markers ("play##ing" → "playing")
        r"##", "", answer
    )
    return answer


# Refine the extracted answer to focus on the core entity for entity-type questions.
def tighten_answer(answer, question):

    q = question.lower()
    if q.startswith(("identify", "which", "what")):
        answer = re.sub(r"^(the|a|an)\s+", "", answer, flags=re.IGNORECASE)
        for splitter in [" is ", " that ", " which ", " who "]:
            if splitter in answer.lower():
                answer = answer.split(splitter)[0].strip()
    return answer


def is_descriptive_answer(answer):
    """Use spaCy Part-of-Speech tagging to check if an answer is purely descriptive (adjective-only)."""
    nlp = get_spacy_nlp()
    doc = nlp(answer)
    noun_count = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
    adj_count = sum(1 for t in doc if t.pos_ == "ADJ")
    return noun_count == 0 and adj_count > 0


def validate_answer(question, answer, context, score):

    if score < 0.1:  # Reject answers with very low confidence
        return False, "low_confidence"
    if not answer:  # Reject if no answer was extracted
        return False, "empty_answer"
    # Reject overly long answers (likely not a precise answer)
    if len(answer.split()) > 20:
        return False, "answer_too_long"
    # Ensure the extracted answer actually exists in the context
    if answer.lower() not in context.lower():
        return False, "not_in_context"
    # For entity-based questions, reject answers that are only descriptive
    if question.lower().startswith(("which", "what", "who")):
        if is_descriptive_answer(answer):
            return False, "descriptive_not_entity"

    return True, "ok"


# Use spacy sentence segmenter to extract the exact sentence containing the answer.
def extract_evidence(answer, context, max_chars=220):

    if not answer:
        return "No answer provided for evidence extraction."

    nlp = get_spacy_nlp()
    doc = nlp(context)
    # Search for the sentence containing the extracted answer
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if answer.lower() in sent_text.lower():
            if len(sent_text) > max_chars:
                return sent_text[:max_chars].rsplit(" ", 1)[0] + "..."
            return sent_text

    return "Relevant evidence sentence not found."


# Split text into word chunks small enough to safely fit BART's 1024-token limit.
def _chunk_text(text, max_words=350):
    # Split the text into a list of words
    words = text.split()

    chunks = []

    # Take 'max_words' words at a time
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


def _summarize_chunk(pipeline_instance, text):
    word_count = len(text.split())
    max_len = min(120, max(20, word_count))  # Set the maximum summary length
    min_len = min(40, max(5, word_count // 3))  # Set the minimum summary length

    # Generate the summary using the BART pipeline
    return pipeline_instance(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,  # Truncate input if it exceeds BART's limit
    )[0]["summary_text"]


def summarize_context(context, max_words_per_chunk=350):

    if not context or len(context.strip()) == 0:
        return "Context is empty."

    try:
        # Load the BART summarization model
        pipeline_instance = load_summary_pipeline()
        word_count = len(context.split())

        # If the text is short enough, summarize it directly
        if word_count <= max_words_per_chunk:
            return _summarize_chunk(pipeline_instance, context)

        # Split the long transcript into smaller chunks
        chunks = _chunk_text(context, max_words_per_chunk)
        chunk_summaries = [
            _summarize_chunk(pipeline_instance, chunk) for chunk in chunks
        ]
        combined = " ".join(chunk_summaries)

        # if they're still too long recurse
        if len(combined.split()) > max_words_per_chunk:
            return summarize_context(combined, max_words_per_chunk)

        return _summarize_chunk(pipeline_instance, combined)
    except Exception as e:
        return f"Summarization failed: {e}"


# Core QA Pipeline

# Extract answers from context using an ensemble of QA models with weighted voting


def reader_qa(question, context):
    # if the question or transcript is empty
    if not question.strip() or not context.strip():
        return {
            "status": "REJECTED",
            "reason": "Question or context is empty.",
            "message": "Question or context is empty.",
        }

    pipelines = load_qa_pipelines()
    answers = []

    for qa, tag in pipelines:
        try:
            result = qa(
                question=question,
                context=context,
                handle_impossible_answer=True,
                max_answer_len=50,
                max_seq_len=512,
                doc_stride=128,
            )

            # Extract the predicted answer
            ans_text = result.get("answer", "")
            if ans_text and ans_text.strip():
                cleaned = clean_answer(ans_text)
                tightened = tighten_answer(cleaned, question)
                if tightened.strip():
                    answers.append(
                        {
                            "answer": tightened,
                            "score": result.get("score", 0.0),
                            "model": tag,
                            "weighted_score": result.get("score", 0.0)
                            * MODEL_WEIGHTS.get(tag, 1.0),
                        }
                    )
        except Exception as e:
            warnings.warn(f"Inference failed for model {tag}: {e}")
            continue

    if not answers:
        return {
            "status": "REJECTED",
            "reason": "No answer could be extracted from the context.",
            "message": "No answer could be extracted from the context.",
        }

    def normalize_key(text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    candidate_scores = {}  # Store the total weighted score of each unique answer
    candidate_display = {}
    for a in answers:
        key = normalize_key(a["answer"])
        candidate_scores[key] = candidate_scores.get(key, 0.0) + a["weighted_score"]
        if key not in candidate_display or a["score"] > candidate_display[key][1]:
            candidate_display[key] = (a["answer"], a["score"])

    best_key = max(candidate_scores, key=candidate_scores.get)
    best_answer = candidate_display[best_key][0]

    supporting_scores = []
    # Collect the confidence scores of models that predicted the winning answer
    for a in answers:
        if normalize_key(a["answer"]) == best_key:
            supporting_scores.append(a["score"])

    # Take the highest confidence score
    if supporting_scores:
        confidence = max(supporting_scores)
    else:
        confidence = 0.0

    is_valid, reason = validate_answer(question, best_answer, context, confidence)
    if not is_valid:
        reason_messages = {
            "low_confidence": "The models could not find a confident enough answer in the context.",
            "empty_answer": "No answer could be extracted from the context.",
            "answer_too_long": "The extracted answer span was too long to be a reliable, specific answer.",
            "not_in_context": "The extracted answer text could not be verified against the context.",
            "descriptive_not_entity": (
                "The context describes characteristics but does not "
                "explicitly state the entity requested."
            ),
        }
        msg = reason_messages.get(reason, "No reliable answer could be extracted.")
        return {
            "status": "NOT_EXPLICIT"
            if reason == "descriptive_not_entity"
            else "REJECTED",
            "reason": msg,
            "message": msg,
        }

    return {
        "status": "ANSWERED",
        "answer": best_answer,
        "confidence": round(confidence, 3),
        "evidence": extract_evidence(best_answer, context),
    }
