import html
import os
import re

import streamlit as st

from pipeline import (
    transcribe_audio,
    summarize_context,
    reader_qa,
)

st.set_page_config(
    page_title="ListenIQ",
    layout="wide",
)

st.markdown(
    """
<style>
    .title-container {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
    }
    .subtitle-text { color: #94a3b8; margin-bottom: 1.5rem; }
    .card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .card-header { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.6rem; }
    .badge {
        padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: var(--bg); color: var(--fg); border: 1px solid var(--fg);
    }
    .highlighted-text {
        background: rgba(59,130,246,0.25); color: #60a5fa; font-weight: 600;
        padding: 1px 5px; border-radius: 4px;
    }
    .evidence-box {
        border-left: 3px solid #8b5cf6; background: rgba(139,92,246,0.05);
        padding: 0.6rem 1rem; border-radius: 0 6px 6px 0; font-style: italic; color: #cbd5e1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Status -> (color, label) used for both the badge and the card's left border
STATUS_STYLE = {
    "ANSWERED": ("#10b981", "Confidence"),
    "NOT_EXPLICIT": ("#ef4444", "Not Explicitly Stated"),
    "REJECTED": ("#f59e0b", "No Answer Found"),
}


def badge(status, confidence=None):
    color, label = STATUS_STYLE[status]
    text = f"{label}: {int(confidence * 100)}%" if status == "ANSWERED" else label
    return f'<span class="badge" style="--bg:{color}1a; --fg:{color};">{text}</span>'


def render_qa_result(question, result, compact=False):
    """Render one Q&A result. compact=True is used for the history list."""
    color = STATUS_STYLE[result["status"]][0]
    if result["status"] == "ANSWERED":
        body = (
            f'<div style="font-size:{"1rem" if compact else "1.3rem"}; font-weight:700; color:#f8fafc;">'
            f"{html.escape(result['answer'])}</div>"
        )
        if not compact:
            body += (
                f'<div style="font-size:0.8rem; color:#94a3b8; margin-top:0.6rem;">SUPPORTING EVIDENCE:</div>'
                f'<div class="evidence-box">"{html.escape(result["evidence"])}"</div>'
            )
    else:
        msg = result.get("message") or result.get(
            "reason", "No reliable answer could be extracted."
        )
        body = (
            f'<div style="color:#cbd5e1; font-size:0.9rem;">{html.escape(msg)}</div>'
            if not compact
            else ""
        )

    confidence = result.get("confidence")
    st.markdown(
        f'<div class="card" style="border-left:4px solid {color};">'
        f'  <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">'
        f'    <span style="font-size:0.85rem; color:#94a3b8;">Q: {html.escape(question)}</span>'
        f"    {badge(result['status'], confidence)}"
        f"  </div>"
        f"  {body}"
        f"</div>",
        unsafe_allow_html=True,
    )


def highlight_answer_in_context(context, answer):

    escaped_context = html.escape(context)
    if not answer:
        return escaped_context.replace("\n", "<br>")
    escaped_answer = html.escape(answer)
    pattern = re.compile(rf"({re.escape(escaped_answer)})", re.IGNORECASE)
    highlighted = pattern.sub(
        r'<span class="highlighted-text">\1</span>', escaped_context
    )
    return highlighted.replace("\n", "<br>")


# State Initialization
defaults = {
    "context": "",
    "summary": "",
    "qa_history": [],
    "last_answer": "",
    "transcribed": False,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# Header
header_col, button_col = st.columns([5, 1])
with header_col:
    st.markdown('<div class="title-container">ListenIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-text">Audio Intelligence — transcription, summarization, and Q&A.</div>',
        unsafe_allow_html=True,
    )
with button_col:
    if st.button(" Clear Workspace"):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

col1, col2 = st.columns([1.1, 0.9], gap="large")

# Column 1: Audio input + transcript
with col1:
    st.markdown(
        '<div class="card-header">Audio / Video Source</div>', unsafe_allow_html=True
    )
    input_method = st.radio(
        "Select Source:",
        ["Upload Audio/Video File", "Record Live Microphone"],
        horizontal=True,
    )

    AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4"}
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    audio_bytes, filename_hint, audio_format, is_video = None, None, "audio/wav", False
    if input_method == "Upload Audio/Video File":
        uploaded = st.file_uploader(
            "Upload wav, mp3, m4a, mp4, mov, mkv, avi, or webm:",
            type=["wav", "mp3", "m4a", "mp4", "mov", "mkv", "avi", "webm"],
        )
        if uploaded is not None:
            audio_bytes = uploaded.read()
            ext = os.path.splitext(uploaded.name)[1].lower()
            is_video = ext in VIDEO_EXTS
            audio_format = AUDIO_MIME.get(ext, "audio/wav")
            filename_hint = uploaded.name
    else:
        recorded = st.audio_input("Record your lecture or meeting:")
        if recorded is not None:
            audio_bytes = recorded.read()
            filename_hint = "recording.wav"

    if audio_bytes is not None:
        if is_video:
            st.video(audio_bytes)
        else:
            st.audio(audio_bytes, format=audio_format)

        if st.button("Transcribe & Analyze Audio", type="primary"):
            with st.spinner("Extracting audio and transcribing with Whisper..."):
                try:
                    transcript = transcribe_audio(
                        audio_bytes, filename_hint=filename_hint
                    )
                    if transcript.strip():
                        st.session_state.update(
                            {
                                "context": transcript,
                                "transcribed": True,
                                "summary": summarize_context(transcript),
                                "qa_history": [],
                                "last_answer": "",
                            }
                        )
                        st.success("Audio transcribed and analyzed successfully!")
                    else:
                        st.error("No speech detected in the audio.")
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
            st.rerun()

    if st.session_state["transcribed"] and st.session_state["context"]:
        st.markdown(
            '<div class="card-header">📄 Transcribed Speech Text</div>',
            unsafe_allow_html=True,
        )
        highlighted = highlight_answer_in_context(
            st.session_state["context"], st.session_state["last_answer"]
        )
        st.markdown(
            f'<div class="card" style="max-height:400px; overflow-y:auto; white-space:pre-wrap; line-height:1.6;">{highlighted}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Upload or record audio, then click the transcribe button to see transcription and analysis."
        )

# Column 2: Summary, Q&A
with col2:
    if st.session_state["transcribed"] and st.session_state["context"]:
        st.markdown(
            '<div class="card-header">AI Intelligence Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-header" style="font-size:0.95rem; color:#94a3b8;">📝 Summary</div>',
            unsafe_allow_html=True,
        )

        summary_text = st.session_state["summary"] or ""
        if summary_text.startswith("Summarization failed:") or summary_text.startswith(
            "Could not load summarization model"
        ):
            st.error(summary_text)
        else:
            st.markdown(
                f'<div class="card">{html.escape(summary_text) or "No summary available."}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="card-header">💬 Chat with the Audio Context</div>',
            unsafe_allow_html=True,
        )
        question = st.text_input(
            "Ask a question about the audio transcription:",
            placeholder="e.g., What deadline was discussed? Who will present next?",
            key="question_input",
        )
        if st.button("Extract Answer", type="secondary"):
            if question.strip():
                with st.spinner("Extracting answer from transcription..."):
                    result = reader_qa(question, st.session_state["context"])
                if result["status"] == "ANSWERED":
                    st.session_state["last_answer"] = result["answer"]
                st.session_state["qa_history"].insert(
                    0, {"question": question, "result": result}
                )
                st.rerun()
            else:
                st.warning("Please enter a question.")

        if st.session_state["qa_history"]:
            latest = st.session_state["qa_history"][0]
            st.markdown("### 🔍 Current Answer")
            render_qa_result(latest["question"], latest["result"])

            if len(st.session_state["qa_history"]) > 1:
                st.markdown("### ⏳ Previous Questions")
                for item in st.session_state["qa_history"][1:]:
                    render_qa_result(item["question"], item["result"], compact=True)
    else:
        st.write("---")
        st.write("Waiting for audio transcription to activate the Q&A panel...")
