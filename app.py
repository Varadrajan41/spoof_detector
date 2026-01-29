import base64
import io
import time
import os
import math
from typing import Dict, Any, Tuple, Optional, List
from flask import render_template

from flask import Flask, request, jsonify

import torch
import torch.nn.functional as F
from pydub import AudioSegment

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = Flask(__name__)

# -------------------- Hackathon Spec Controls --------------------
API_KEY = os.getenv("API_KEY", "sk_test_123456789")  # set this in deployment env
SUPPORTED_LANGS = {"tamil", "english", "hindi", "malayalam", "telugu"}

# -------------------- Audio / Inference Params --------------------
TARGET_SR = 16000
FIXED_NUM_SAMPLES = 64600  # ~4.04s at 16kHz

DEFAULT_THRESHOLD = float(os.getenv("SPOOF_THRESHOLD", "0.3"))
DEFAULT_TOPK = int(os.getenv("ROBUST_TOPK", "3"))
HOP_SECONDS = float(os.getenv("ROBUST_HOP_SECONDS", "2.0"))

# Auto-robust behavior: if clip longer than window, slide windows
AUTO_ROBUST = os.getenv("AUTO_ROBUST", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

# Model
MODEL_ID = os.getenv("MODEL_ID", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification")


# Optional: include debug fields in response (keep OFF for hackathon)
INCLUDE_DEBUG = os.getenv("INCLUDE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

# -------------------- Auth --------------------
@app.before_request
def require_api_key():
    # enforce only for the hackathon endpoint (and optionally others)
    if request.path == "/api/voice-detection":
        key = request.headers.get("x-api-key", "")
        if not key or key != API_KEY:
            return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 401

# -------------------- Model --------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_extractor(device: torch.device):
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return model, extractor

MODEL_DEVICE = get_device()
MODEL, FEATURE_EXTRACTOR = load_model_and_extractor(MODEL_DEVICE)

def _resolve_label_indices(model) -> Tuple[int, int]:
    """
    Resolve bonafide vs spoof indices robustly from config.
    Falls back to (0,1).
    """
    id2label = getattr(model.config, "id2label", None) or {}
    norm = {}
    for k, v in id2label.items():
        try:
            kk = int(k)
        except Exception:
            continue
        norm[kk] = str(v).lower()

    spoof_tokens = ("spoof", "fake", "ai", "synthetic", "deepfake")
    bona_tokens = ("bonafide", "bona-fide", "real", "human", "genuine")

    spoof_idx = None
    bona_idx = None

    for i, lab in norm.items():
        if any(t in lab for t in spoof_tokens):
            spoof_idx = i
        if any(t in lab for t in bona_tokens):
            bona_idx = i

    if bona_idx is None or spoof_idx is None:
        return 0, 1
    return int(bona_idx), int(spoof_idx)

BONA_IDX, SPOOF_IDX = _resolve_label_indices(MODEL)

def infer_probs_from_wav_batch(wav_batch: torch.Tensor) -> torch.Tensor:
    """
    wav_batch: [B, T] float32 waveform, 16kHz mono, values ~[-1,1]
    returns probs: [B, 2] -> [p_bonafide, p_spoof]
    """
    if wav_batch.dim() != 2:
        raise ValueError(f"Expected wav_batch [B,T], got shape {tuple(wav_batch.shape)}")

    wav_list = [w.detach().cpu().numpy() for w in wav_batch]

    inputs = FEATURE_EXTRACTOR(
        wav_list,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = MODEL(**inputs)
        probs_full = torch.softmax(out.logits, dim=-1)  # [B, C]

    # map to [bonafide, spoof]
    pb = probs_full[:, BONA_IDX].unsqueeze(1)
    ps = probs_full[:, SPOOF_IDX].unsqueeze(1)
    probs = torch.cat([pb, ps], dim=1)  # [B,2]
    return probs

# -------------------- Base64 / Audio Decode --------------------
def clean_base64(s: str) -> str:
    """
    Handles:
      - whitespace/newlines
      - data URI prefixes like: data:audio/mp3;base64,AAAA...
    """
    if not isinstance(s, str):
        raise ValueError("audioBase64 must be a string")

    s = s.strip()

    # If it’s a data URI, take payload after the first comma
    if s.lower().startswith("data:") and "base64," in s.lower():
        s = s.split(",", 1)[1]

    # remove whitespace/newlines
    s = "".join(s.split())
    return s

def mp3_bytes_to_wav_16k_mono(audio_bytes: bytes) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Hackathon input is ALWAYS mp3.
    We force format='mp3' to prevent ffmpeg/pydub guessing wrong (e.g., treating MP3 as WAV).
    """
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    orig_channels = seg.channels
    orig_sr = seg.frame_rate
    orig_width = seg.sample_width

    # internal normalization for model
    seg = seg.set_channels(1).set_frame_rate(TARGET_SR)

    samples = seg.get_array_of_samples()
    max_int = float(2 ** (8 * orig_width - 1)) if orig_width else 32768.0
    wav = torch.tensor(samples, dtype=torch.float32) / max_int
    wav = wav.unsqueeze(0)  # [1, T]

    meta = {
        "input_format": "mp3",
        "orig_sample_rate": orig_sr,
        "orig_channels": orig_channels,
        "orig_sample_width_bytes": orig_width,
        "final_sample_rate": TARGET_SR,
        "final_channels": 1,
        "num_samples": int(wav.shape[1]),
        "duration_sec": float(wav.shape[1]) / TARGET_SR,
    }
    return wav, meta

# -------------------- Windowing helpers --------------------
def pad_or_truncate(wav_1xt: torch.Tensor, target_len: int, pad_mode: str = "repeat") -> Tuple[torch.Tensor, str]:
    t = int(wav_1xt.shape[1])
    if t <= 0:
        return torch.zeros((1, target_len), dtype=wav_1xt.dtype), "empty_zero_padded"

    if t < target_len:
        if pad_mode == "repeat":
            reps = int(math.ceil(target_len / t))
            wav_rep = wav_1xt.repeat(1, reps)[:, :target_len]
            return wav_rep, "repeat_padded"
        return F.pad(wav_1xt, (0, target_len - t)), "zero_padded"

    if t > target_len:
        return wav_1xt[:, :target_len], "truncated"

    return wav_1xt, "unchanged"

def get_windows(wav_1xt: torch.Tensor, win_len: int, hop_len: int) -> torch.Tensor:
    """
    wav_1xt: [1, T] -> windows: [N, win_len]
    """
    assert wav_1xt.dim() == 2 and wav_1xt.size(0) == 1
    T = wav_1xt.size(1)

    if T <= win_len:
        w, _ = pad_or_truncate(wav_1xt, win_len, pad_mode="repeat")
        return w.squeeze(0).unsqueeze(0)

    n = 1 + (T - win_len) // hop_len
    if (T - win_len) % hop_len != 0:
        n += 1

    chunks = []
    for i in range(n):
        start = i * hop_len
        end = start + win_len
        chunk = wav_1xt[:, start:end]
        chunk, _ = pad_or_truncate(chunk, win_len, pad_mode="repeat")
        chunks.append(chunk.squeeze(0))
    return torch.stack(chunks, dim=0)

def topk_mean_probs(p_bonafide_list: List[float], p_spoof_list: List[float], k: int) -> Tuple[float, float, Dict[str, Any]]:
    n = len(p_spoof_list)
    if n == 0:
        return 0.0, 0.0, {"num_windows": 0}

    k = max(1, min(int(k), n))
    top_idx = sorted(range(n), key=lambda i: p_spoof_list[i], reverse=True)[:k]

    ps = sum(p_spoof_list[i] for i in top_idx) / k
    pb = sum(p_bonafide_list[i] for i in top_idx) / k

    s = pb + ps
    if s > 0:
        pb /= s
        ps /= s

    stats = {
        "num_windows": n,
        "topk": k,
        "p_spoof_max": float(max(p_spoof_list)),
        "p_spoof_mean": float(sum(p_spoof_list) / n),
    }
    return float(pb), float(ps), stats

# -------------------- Explanation (short, spec-friendly) --------------------
def short_explanation(classification: str, p_spoof: float, threshold: float) -> str:
    if classification == "AI_GENERATED":
        return f"Detected synthetic-voice artifacts; spoof score {p_spoof:.2f} exceeded threshold {threshold:.2f}."
    return f"No strong synthetic artifacts; spoof score {p_spoof:.2f} below threshold {threshold:.2f}."

# -------------------- Routes --------------------


@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(MODEL_DEVICE),
        "model_id": MODEL_ID,
    })

@app.post("/api/voice-detection")
def voice_detection():
    t0 = time.time()

    payload = request.get_json(force=True, silent=True) or {}

    # Required fields per spec
    language = payload.get("language", "")
    audio_format = payload.get("audioFormat", "")
    audio_b64 = payload.get("audioBase64", "")

    # Basic validation
    if not isinstance(language, str) or not language.strip():
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400
    lang_norm = language.strip().lower()
    if lang_norm not in SUPPORTED_LANGS:
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400

    if not isinstance(audio_format, str) or audio_format.strip().lower() != "mp3":
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400

    if not isinstance(audio_b64, str) or not audio_b64.strip():
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400

    # Optional knobs (won't break spec if ignored by evaluator)
    # If present, they can help your own testing.
    topk = DEFAULT_TOPK
    if payload.get("topk") is not None:
        try:
            topk = int(payload.get("topk"))
        except Exception:
            topk = DEFAULT_TOPK

    # Decode base64 -> MP3 bytes
    try:
        audio_b64_clean = clean_base64(audio_b64)
        audio_bytes = base64.b64decode(audio_b64_clean)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400

    # Decode MP3 -> waveform
    try:
        wav, audio_meta = mp3_bytes_to_wav_16k_mono(audio_bytes)
        wav_stats = {
            "min": float(wav.min().item()),
            "max": float(wav.max().item()),
            "mean": float(wav.mean().item()),
            "std": float(wav.std().item()),
        }
    except Exception:
        # keep message generic for hackathon
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 400

    # Too-short guard (your old logic, but hackathon wants only HUMAN/AI_GENERATED;
    # we’ll still return HUMAN with low confidence rather than UNKNOWN)
    if audio_meta["duration_sec"] < 0.5:
        resp = {
            "status": "success",
            "language": language,
            "classification": "HUMAN",
            "confidenceScore": 0.50,
            "explanation": "Audio too short for reliable decision; defaulting to HUMAN with low confidence.",
        }
        return jsonify(resp), 200

    analyzed_sec = FIXED_NUM_SAMPLES / TARGET_SR
    do_sliding = bool(AUTO_ROBUST and audio_meta["duration_sec"] > analyzed_sec + 0.1)

    try:
        if not do_sliding:
            wav_fixed, pad_action = pad_or_truncate(wav, FIXED_NUM_SAMPLES, pad_mode="repeat")
            probs = infer_probs_from_wav_batch(wav_fixed)  # [1,2]
            p_bonafide = float(probs[0, 0].item())
            p_spoof = float(probs[0, 1].item())
            mode = "fast"
            window_stats = None
        else:
            hop_len = int(HOP_SECONDS * TARGET_SR)
            win_len = FIXED_NUM_SAMPLES
            windows = get_windows(wav, win_len=win_len, hop_len=hop_len)  # [N, win_len]
            probs = infer_probs_from_wav_batch(windows)  # [N,2]

            pb_list = probs[:, 0].detach().cpu().tolist()
            ps_list = probs[:, 1].detach().cpu().tolist()

            p_bonafide, p_spoof, stats = topk_mean_probs(pb_list, ps_list, k=topk)
            mode = "robust"
            window_stats = {
                **stats,
                "window_sec": float(win_len / TARGET_SR),
                "hop_sec": float(hop_len / TARGET_SR),
                "aggregation": f"topk_mean(k={max(1, min(int(topk), len(ps_list)))})",
            }
            pad_action = "sliding_windows"
    except Exception:
        return jsonify({"status": "error", "message": "Invalid API key or malformed request"}), 500

    # Decision
    if p_spoof >= DEFAULT_THRESHOLD:
        classification = "AI_GENERATED"
        confidence = p_spoof
    else:
        classification = "HUMAN"
        confidence = p_bonafide

    resp = {
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": round(float(confidence), 4),
        "explanation": short_explanation(classification, p_spoof, DEFAULT_THRESHOLD),
    }

    # Optional debug (keep OFF for submission unless allowed)
    if INCLUDE_DEBUG:
        resp["_debug"] = {
            "model_id": MODEL_ID,
            "scores": {"p_bonafide": round(float(p_bonafide), 6), "p_spoof": round(float(p_spoof), 6)},
            "mode": mode,
            "audio_meta": audio_meta,
            "waveform_stats": wav_stats,
            "windowing": window_stats,
            "processing_ms": int((time.time() - t0) * 1000),
            "threshold": DEFAULT_THRESHOLD,
            "label_indices": {"bonafide": BONA_IDX, "spoof": SPOOF_IDX},
        }

    return jsonify(resp), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    # use_reloader=False to avoid double-loading HF models
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
