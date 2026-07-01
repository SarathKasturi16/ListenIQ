# -*- coding: utf-8 -*-
"""
Core NLP Pipeline for VoxMind.
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


def get_spacy_nlp():
    """Lazily load and cache the spaCy model, auto-downloading it if missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model package not downloaded yet (common on a fresh install
            # where 'python -m spacy download en_core_web_sm' was skipped).
            # Download it programmatically instead of crashing.
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
    """Lazily load and cache QA models as pipelines."""
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
                    "question-answering",
                    model=model_name,
                    tokenizer=AutoTokenizer.from_pretrained(model_name),
                    device=device,
                )
                _qa_pipelines.append((qa, model_tag))
            except Exception as e:
                warnings.warn(f"Could not load QA model '{model_name}': {e}")

        if not _qa_pipelines:
            raise RuntimeError("No QA pipelines could be loaded.")
    return _qa_pipelines


def load_summary_pipeline():
    """Lazily load and cache the BART summarization pipeline."""
    global _summary_pipeline
    if _summary_pipeline is None:
        try:
            _summary_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
            )
        except Exception as e:
            # Re-raise instead of swallowing: BART-large-CNN is ~1.6GB, so
            # failures here are almost always a first-time download issue
            # (no internet / blocked host) or an out-of-memory error. Hiding
            # the real exception behind "not available" makes this
            # impossible to diagnose.
            raise RuntimeError(
                f"Could not load summarization model 'facebook/bart-large-cnn': {e}"
            ) from e
    return _summary_pipeline


def load_speech_pipeline():
    """
    Lazily load and cache the Whisper-Tiny ASR pipeline.

    IMPORTANT: chunk_length_s/stride_length_s are required for anything longer
    than ~30s of audio. Whisper's encoder processes fixed 30s windows; without
    chunking, the HF pipeline silently transcribes only the first 30 seconds
    of any longer file and returns no error. Since this app targets meeting
    and lecture audio (typically minutes long), chunking is mandatory, not
    optional.
    """
    global _speech_pipeline
    if _speech_pipeline is None:
        try:
            _speech_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device,
                chunk_length_s=30,
                stride_length_s=5,
            )
        except Exception as e:
            raise RuntimeError(f"Could not load Whisper model: {e}")
    return _speech_pipeline


# --- Audio & Speech functions ---


def load_audio_waveform(audio_source, target_sr=16000):
    """
    Decode any common audio OR video container (wav/mp3/m4a/mp4/mov/mkv/etc.)
    into a mono 16kHz float32 numpy array using pydub (which wraps ffmpeg).

    `audio_source` can be a file path OR an in-memory file-like object
    (e.g. io.BytesIO) — pydub's AudioSegment.from_file accepts both, which
    lets callers avoid writing anything to disk.

    soundfile/librosa only natively read wav/flac/ogg; everything else
    (compressed audio, and any video container) needs ffmpeg to demux/decode.
    ffmpeg auto-detects the container format, so passing a video file here
    works the same way as an audio file: ffmpeg extracts just the audio
    stream and discards the video stream. No separate extraction step or
    extra model is needed for video support — Whisper only ever sees audio.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError(
            "pydub is required for audio decoding. Install it with "
            "'pip install pydub' (and ensure ffmpeg is installed and on PATH)."
        ) from e

    try:
        audio = AudioSegment.from_file(audio_source)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode/extract audio. This usually means ffmpeg is "
            f"not installed or not on PATH, or the file has no audio track. "
            f"Install ffmpeg with 'sudo apt-get install ffmpeg' (Linux), "
            f"'brew install ffmpeg' (Mac), or download it from "
            f"https://ffmpeg.org/download.html (Windows). "
            f"Original error: {e}"
        )

    audio = audio.set_frame_rate(target_sr).set_channels(1)

    import numpy as np

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # pydub gives integer PCM samples; normalize to [-1, 1] as Whisper expects
    max_val = float(1 << (8 * audio.sample_width - 1))
    samples /= max_val

    return samples, target_sr


def transcribe_audio(audio_source, filename_hint=None):
    """
    Transcribe speech from an audio OR video source using OpenAI Whisper-Tiny.

    `audio_source` can be a file path, raw bytes, or an in-memory file-like
    object (e.g. io.BytesIO from a Streamlit upload) — no temp file needed.
    `filename_hint` (e.g. "clip.mp4") helps pydub/ffmpeg pick the right
    decoder when `audio_source` is raw bytes/BytesIO with no extension info.
    Video files are supported transparently: load_audio_waveform extracts
    just the audio track via ffmpeg before this function ever sees it.
    """
    if audio_source is None:
        raise ValueError("No audio source provided.")

    if isinstance(audio_source, (bytes, bytearray)):
        import io

        buf = io.BytesIO(audio_source)
        if filename_hint:
            buf.name = filename_hint  # pydub/ffmpeg use this to infer format
        audio_source = buf
    elif isinstance(audio_source, str):
        if not os.path.exists(audio_source):
            raise ValueError(f"Audio file path does not exist: {audio_source}")

    asr = load_speech_pipeline()
    try:
        waveform, sampling_rate = load_audio_waveform(audio_source)
        # Pass a raw waveform dict instead of a file path: this avoids
        # relying on soundfile/librosa's container auto-detection, which is
        # what was failing on mp3 input.
        result = asr(
            {"array": waveform, "sampling_rate": sampling_rate}, return_timestamps=True
        )
        return result.get("text", "").strip()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


# --- Utility Functions ---


def clean_answer(answer):
    """Normalize whitespace and remove subword tokenization artifacts."""
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.sub(r"##", "", answer)
    return answer


def tighten_answer(answer, question):
    """Refine the extracted answer to focus on the core entity for entity-type questions."""
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
    """
    Perform technical, linguistic, and confidence validation on the answer span.
    Returns (is_valid: bool, reason: str). `reason` is only meaningful when
    is_valid is False, and lets callers distinguish *why* validation failed
    instead of always assuming a linguistic rejection.
    """
    if score < 0.1:
        return False, "low_confidence"
    if not answer:
        return False, "empty_answer"
    if len(answer.split()) > 20:
        return False, "answer_too_long"
    if answer.lower() not in context.lower():
        return False, "not_in_context"

    if question.lower().startswith(("which", "what", "who")):
        if is_descriptive_answer(answer):
            return False, "descriptive_not_entity"

    return True, "ok"


def extract_evidence(answer, context, max_chars=220):
    """Use spaCy's sentence segmenter to extract the exact sentence containing the answer."""
    if not answer:
        return "No answer provided for evidence extraction."

    nlp = get_spacy_nlp()
    doc = nlp(context)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if answer.lower() in sent_text.lower():
            if len(sent_text) > max_chars:
                return sent_text[:max_chars].rsplit(" ", 1)[0] + "..."
            return sent_text

    return "Relevant evidence sentence not found."


def _chunk_text(text, max_words=350):
    """Split text into word chunks small enough to safely fit BART's 1024-token limit."""
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _summarize_chunk(pipeline_instance, text):
    """Summarize a single chunk that's already within BART's input limit."""
    word_count = len(text.split())
    max_len = min(120, max(20, word_count))
    min_len = min(40, max(5, word_count // 3))
    return pipeline_instance(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"]


def summarize_context(context, max_words_per_chunk=350):
    """
    Generate a summary of the given context, regardless of its length.

    BART-large-CNN has a hard 1024-token input limit; passing a long
    transcript straight through with truncation=True silently drops
    everything past that point, so a 45-minute lecture only ever gets
    summarized on roughly its first few minutes. To cover the entire
    transcript, long input is split into word-count chunks (well under the
    token limit, since tokens != words), each chunk is summarized
    independently, and the combined chunk-summaries are summarized once
    more into a final, cohesive summary.
    """
    if not context or len(context.strip()) == 0:
        return "Context is empty."

    try:
        pipeline_instance = load_summary_pipeline()
        word_count = len(context.split())

        if word_count <= max_words_per_chunk:
            return _summarize_chunk(pipeline_instance, context)

        chunks = _chunk_text(context, max_words_per_chunk)
        chunk_summaries = [
            _summarize_chunk(pipeline_instance, chunk) for chunk in chunks
        ]
        combined = " ".join(chunk_summaries)

        # The combined chunk-summaries are usually short enough to summarize
        # in one more pass; if they're still too long, recurse.
        if len(combined.split()) > max_words_per_chunk:
            return summarize_context(combined, max_words_per_chunk)

        return _summarize_chunk(pipeline_instance, combined)
    except Exception as e:
        return f"Summarization failed: {e}"


# --- Core QA Pipeline ---


def reader_qa(question, context):
    """
    Extract answers from context using an ensemble of QA models with chunked inference,
    weighted voting, and linguistic validation.
    """
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

    # Aggregate scores using a normalized key (lowercased, punctuation-stripped)
    # so minor casing/punctuation differences between models (e.g. "Tokyo" vs
    # "tokyo,") don't split votes that should reinforce one candidate. The
    # best-scoring original surface form is kept for display.
    def normalize_key(text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    candidate_scores = {}
    candidate_display = {}
    for a in answers:
        key = normalize_key(a["answer"])
        candidate_scores[key] = candidate_scores.get(key, 0.0) + a["weighted_score"]
        if key not in candidate_display or a["score"] > candidate_display[key][1]:
            candidate_display[key] = (a["answer"], a["score"])

    best_key = max(candidate_scores, key=candidate_scores.get)
    best_answer = candidate_display[best_key][0]

    supporting_scores = [
        a["score"] for a in answers if normalize_key(a["answer"]) == best_key
    ]
    confidence = max(supporting_scores) if supporting_scores else 0.0

    is_valid, reason = validate_answer(question, best_answer, context, confidence)
    if not is_valid:
        # Only the linguistic case ("looks like a description, not the
        # named entity the question asked for") gets the "not explicitly
        # stated" framing. Everything else gets its own accurate message —
        # previously all rejection reasons (low confidence, answer too long,
        # answer not actually in the context) were misreported with that
        # same linguistic message, which was misleading.
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
