# import gradio as gr
# import requests

# API_URL = "http://127.0.0.1:5000/detect"

# def detect(audio_path, language, robust, topk):
#     if audio_path is None:
#         return {"error": "No audio provided"}
#     with open(audio_path, "rb") as f:
#         audio_bytes = f.read()

#     files = {"file": (audio_path.split("/")[-1], audio_bytes)}
#     data = {
#         "language": language or "",
#         "robust": str(bool(robust)).lower(),
#         "topk": str(int(topk)),
#     }
#     r = requests.post(API_URL, files=files, data=data, timeout=120)
#     r.raise_for_status()
#     return r.json()

# demo = gr.Interface(
#     fn=detect,
#     inputs=[
#         gr.Audio(sources=["microphone", "upload"], type="filepath"),  # upload + mic in one :contentReference[oaicite:3]{index=3}
#         gr.Dropdown(choices=["", "en", "hi", "ta", "ml", "te"], value="", label="Language (optional)"),
#         gr.Checkbox(value=False, label="Robust mode"),
#         gr.Number(value=3, precision=0, label="Top-K (robust)"),
#     ],
#     outputs=gr.JSON(),
#     title="Spoof Detector (Upload + Microphone)",
# )

# if __name__ == "__main__":
#     demo.launch()


import json
import requests
import gradio as gr

def call_api(audio_path, backend, robust, topk, language):
    if audio_path is None:
        return "No audio provided", {}

    base_url = "http://127.0.0.1:5000" if backend == "Single (app.py)" else "http://127.0.0.1:5001"
    detect_url = base_url.rstrip("/") + "/detect"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "language": language or "en",
        "robust": str(bool(robust)).lower(),
        "topk": str(int(topk)),
    }

    r = requests.post(detect_url, files=files, data=data, timeout=120)
    try:
        out = r.json()
    except Exception:
        out = {"error": "Non-JSON response", "status": r.status_code, "text": r.text}

    summary = f"HTTP {r.status_code} | classification={out.get('classification')} | confidence={out.get('confidence')}"
    return summary, out

with gr.Blocks(title="Spoof Detector UI") as demo:
    gr.Markdown("# 🎙️ Spoof Detector (Single / Ensemble)")

    backend = gr.Radio(choices=["Single (app.py)", "Ensemble (ensemble.py)"], value="Single (app.py)", label="Backend")

    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio input")

    with gr.Row():
        robust = gr.Checkbox(value=False, label="Robust (sliding windows)")
        topk = gr.Slider(1, 10, value=3, step=1, label="TopK (robust)")
        # language = gr.Textbox(value="en", label="Language")
        language = gr.Dropdown(choices=["", "en", "hi", "ta", "ml", "te"], value="", label="Language (optional)")
        # language=gr.Dropdown(choices=["", "en", "hi", "ta", "ml", "te"], value="", label="Language (optional)"),

    btn = gr.Button("Detect")
    summary = gr.Textbox(label="Summary")
    out_json = gr.JSON(label="Response JSON")

    btn.click(fn=call_api, inputs=[audio, backend, robust, topk, language], outputs=[summary, out_json])

demo.launch(server_name="0.0.0.0", server_port=7860)
