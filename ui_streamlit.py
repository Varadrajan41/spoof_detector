import json
import requests
import streamlit as st

st.set_page_config(page_title="Spoof Detector UI", layout="centered")
st.title("🎙️ Spoof Detector (Single / Ensemble)")

mode = st.radio("Backend", ["Single (app.py)", "Ensemble (ensemble.py)"], horizontal=True)

default_url = "http://127.0.0.1:5000" if mode.startswith("Single") else "http://127.0.0.1:5001"
base_url = st.text_input("Backend base URL", value=default_url)
detect_url = base_url.rstrip("/") + "/detect"
health_url = base_url.rstrip("/") + "/health"

col1, col2 = st.columns(2)
with col1:
    robust = st.checkbox("Robust (sliding windows)", value=False)
with col2:
    topk = st.number_input("TopK (robust)", min_value=1, max_value=10, value=3, step=1)

language = st.text_input("Language (optional)", value="en")

st.divider()
st.subheader("A) Microphone")
audio_rec = st.audio_input("Record from microphone", sample_rate=16000)

st.subheader("B) Upload file")
uploaded = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])

st.divider()

if st.button("🔎 Detect"):
    chosen = audio_rec if audio_rec is not None else uploaded
    if chosen is None:
        st.error("Please record audio or upload a file.")
        st.stop()

    # Play back
    st.audio(chosen)

    files = {"file": ("audio.wav", chosen.getvalue(), "audio/wav")}
    data = {
        "language": language,
        "robust": str(robust).lower(),
        "topk": str(int(topk)),
    }

    try:
        r = requests.post(detect_url, files=files, data=data, timeout=120)
        st.write("HTTP:", r.status_code)
        out = r.json()
        st.json(out)

        st.success(f"Classification: {out.get('classification')} | Confidence: {out.get('confidence')}")
        st.caption(out.get("explanation_text", ""))
    except Exception as e:
        st.error(f"Request failed: {e}")

with st.expander("Health check"):
    try:
        r = requests.get(health_url, timeout=10)
        st.json(r.json())
    except Exception as e:
        st.error(f"Health failed: {e}")
