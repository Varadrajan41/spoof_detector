import base64
import io
import time
import os
import math
from typing import Dict, Any, Tuple, Optional, List

from flask import Flask, request, jsonify

import torch
import torch.nn.functional as F
from pydub import AudioSegment

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = Flask(__name__)

TARGET_SR = 16000
FIXED_NUM_SAMPLES = 64600  # ~4.04s at 16kHz
DEFAULT_THRESHOLD = float(os.getenv("SPOOF_THRESHOLD", "0.3"))

DEFAULT_TOPK = int(os.getenv("ROBUST_TOPK", "3"))
HOP_SECONDS = float(os.getenv("ROBUST_HOP_SECONDS", "2.0"))

# ---- Ensemble model IDs (no HyperMoon) ----
MODEL_A_ID = os.getenv("MODEL_A_ID", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification")
MODEL_B_ID = os.getenv("MODEL_B_ID", "garystafford/wav2vec2-deepfake-voice-detector")

# weights for fusion (must sum ~1; we normalize anyway)
MODEL_A_W = float(os.getenv("MODEL_A_WEIGHT", "0.6"))
MODEL_B_W = float(os.getenv("MODEL_B_WEIGHT", "0.4"))

# -------------------- Device --------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DEVICE = get_device()

# -------------------- Model loading --------------------
def load_one_model(model_id: str, device: torch.device):
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval().to(device)
    return model, extractor

MODEL_A, FE_A = load_one_model(MODEL_A_ID, MODEL_DEVICE)
MODEL_B, FE_B = load_one_model(MODEL_B_ID, MODEL_DEVICE)

def _normalize_weights(w1: float, w2: float) -> Tuple[float, float]:
    s = w1 + w2
    if s <= 0:
        return 0.5, 0.5
    return w1 / s, w2 / s

MODEL_A_W, MODEL_B_W = _normalize_weights(MODEL_A_W, MODEL_B_W)

def get_label_indices(model) -> Tuple[int, int, Dict[str, Any]]:
    """
    Returns (bonafide_index, spoof_index, debug_info) by reading config labels.
    Tries to find labels like fake/spoof/bonafide/real.
    Falls back to (0,1) if ambiguous.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    # id2label keys may be int or str in practice
    norm = {}
    for k, v in id2label.items():
        try:
            kk = int(k)
        except Exception:
            kk = k
        norm[kk] = str(v).lower()

    # search for spoof/fake first
    spoof_candidates = {"fake", "spoof", "ai", "synthetic", "deepfake"}
    bona_candidates = {"real", "bonafide", "human", "genuine"}

    spoof_idx = None
    bona_idx = None

    for k, v in norm.items():
        if any(tok in v for tok in spoof_candidates):
            spoof_idx = k
        if any(tok in v for tok in bona_candidates):
            bona_idx = k

    # if still ambiguous, try label2id
    label2id = getattr(model.config, "label2id", None) or {}
    label2id_norm = {str(k).lower(): int(v) for k, v in label2id.items()} if isinstance(label2id, dict) else {}

    if spoof_idx is None:
        for key in ["fake", "spoof", "ai", "synthetic", "deepfake"]:
            if key in label2id_norm:
                spoof_idx = label2id_norm[key]
                break

    if bona_idx is None:
        for key in ["real", "bonafide", "human", "genuine"]:
            if key in label2id_norm:
                bona_idx = label2id_norm[key]
                break

    # final fallback
    if bona_idx is None or spoof_idx is None:
        bona_idx, spoof_idx = 0, 1

    dbg = {
        "id2label": id2label,
        "label2id": label2id,
        "resolved_indices": {"bonafide": bona_idx, "spoof": spoof_idx},
    }
    return int(bona_idx), int(spoof_idx), dbg

BONA_A, SPOOF_A, LABELDBG_A = get_label_indices(MODEL_A)
BONA_B, SPOOF_B, LABELDBG_B = get_label_indices(MODEL_B)

def infer_probs_one(model, extractor, wav_batch: torch.Tensor, bona_idx: int, spoof_idx: int) -> Tuple[List[float], List[float]]:
    """
    wav_batch: [B, T] float32 waveform, 16kHz mono, ~[-1,1]
    returns pb_list, ps_list (python lists) aligned with batch order
    """
    if wav_batch.dim() != 2:
        raise ValueError(f"Expected wav_batch [B,T], got {tuple(wav_batch.shape)}")

    wav_list = [w.detach().cpu().numpy() for w in wav_batch]
    inputs = extractor(wav_list, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)  # [B, C]

    pb = probs[:, bona_idx].detach().cpu().tolist()
    ps = probs[:, spoof_idx].detach().cpu().tolist()
    return pb, ps

def infer_probs_ensemble(wav_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Returns:
      fused_probs: [B, 2] where [:,0]=p_bonafide, [:,1]=p_spoof
      debug: per-model summary
    """
    pb_a, ps_a = infer_probs_one(MODEL_A, FE_A, wav_batch, BONA_A, SPOOF_A)
    pb_b, ps_b = infer_probs_one(MODEL_B, FE_B, wav_batch, BONA_B, SPOOF_B)

    # fuse spoof + bonafide with weights (then renormalize)
    pb_f = []
    ps_f = []
    for i in range(len(pb_a)):
        ps = MODEL_A_W * float(ps_a[i]) + MODEL_B_W * float(ps_b[i])
        pb = MODEL_A_W * float(pb_a[i]) + MODEL_B_W * float(pb_b[i])
        s = pb + ps
        if s > 0:
            pb /= s
            ps /= s
        pb_f.append(pb)
        ps_f.append(ps)

    fused = torch.tensor(list(zip(pb_f, ps_f)), dtype=torch.float32)  # [B,2]

    debug = {
        "weights": {"model_a": MODEL_A_W, "model_b": MODEL_B_W},
        "model_a": {
            "id": MODEL_A_ID,
            "bonafide_idx": BONA_A,
            "spoof_idx": SPOOF_A,
            "p_spoof_max": float(max(ps_a)) if ps_a else None,
            "p_spoof_mean": float(sum(ps_a) / len(ps_a)) if ps_a else None,
        },
        "model_b": {
            "id": MODEL_B_ID,
            "bonafide_idx": BONA_B,
            "spoof_idx": SPOOF_B,
            "p_spoof_max": float(max(ps_b)) if ps_b else None,
            "p_spoof_mean": float(sum(ps_b) / len(ps_b)) if ps_b else None,
        },
    }
    return fused, debug

# -------------------- Audio utils --------------------
def sniff_audio_format(data: bytes) -> Optional[str]:
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 3 and data[0:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    return None

def bytes_to_wav_16k_mono(audio_bytes: bytes, audio_format: Optional[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    fmt = audio_format or sniff_audio_format(audio_bytes)

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    orig_channels = seg.channels
    orig_sr = seg.frame_rate
    orig_width = seg.sample_width

    seg = seg.set_channels(1).set_frame_rate(TARGET_SR)

    samples = seg.get_array_of_samples()
    max_int = float(2 ** (8 * orig_width - 1))
    wav = torch.tensor(samples, dtype=torch.float32) / max_int
    wav = wav.unsqueeze(0)  # [1, T]

    meta = {
        "input_format": fmt or "unknown",
        "orig_sample_rate": orig_sr,
        "orig_channels": orig_channels,
        "orig_sample_width_bytes": orig_width,
        "final_sample_rate": TARGET_SR,
        "final_channels": 1,
        "num_samples": int(wav.shape[1]),
        "duration_sec": float(wav.shape[1]) / TARGET_SR,
    }
    return wav, meta

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

# -------------------- Robust helpers --------------------
def parse_bool(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def get_windows(wav_1xt: torch.Tensor, win_len: int, hop_len: int) -> torch.Tensor:
    assert wav_1xt.dim() == 2 and wav_1xt.size(0) == 1
    T = wav_1xt.size(1)

    if T <= win_len:
        w, _ = pad_or_truncate(wav_1xt, win_len, pad_mode="repeat")
        return w.squeeze(0).unsqueeze(0)  # [1, win_len]

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
    return torch.stack(chunks, dim=0)  # [N, win_len]

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

# -------------------- Explanation --------------------
def make_explanation_text(
    classification: str,
    confidence: float,
    p_bonafide: float,
    p_spoof: float,
    threshold: float,
    audio_meta: Dict[str, Any],
    mode: str,
    pad_action: str,
    window_stats: Optional[Dict[str, Any]] = None,
) -> str:
    dur = audio_meta.get("duration_sec", None)
    in_fmt = audio_meta.get("input_format", "unknown")
    orig_sr = audio_meta.get("orig_sample_rate", "unknown")
    final_sr = audio_meta.get("final_sample_rate", TARGET_SR)
    analyzed_sec = FIXED_NUM_SAMPLES / TARGET_SR

    parts = []
    if isinstance(dur, (int, float)):
        parts.append(f"Processed a {dur:.2f}s {in_fmt.upper()} clip (orig {orig_sr} Hz) and converted it to {final_sr} Hz mono.")
    else:
        parts.append(f"Processed a {in_fmt.upper()} clip and converted it to {final_sr} Hz mono.")

    if mode == "fast":
        parts.append(f"Mode=fast: analyzed ~{analyzed_sec:.2f}s (pad/truncate: {pad_action}).")
    else:
        if window_stats:
            parts.append(
                f"Mode=robust: analyzed {window_stats.get('num_windows')} windows "
                f"(window={window_stats.get('window_sec', analyzed_sec):.2f}s, hop={window_stats.get('hop_sec', 2.0):.2f}s) "
                f"using top-{window_stats.get('topk', DEFAULT_TOPK)} mean aggregation."
            )
        else:
            parts.append("Mode=robust: used sliding windows and top-K mean aggregation.")

    parts.append(f"Ensemble scores: spoof={p_spoof:.6f}, bonafide={p_bonafide:.6f}; threshold={threshold:.2f}.")
    parts.append(f"Final classification: {classification} (confidence={confidence:.4f}).")
    return " ".join(parts)

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(MODEL_DEVICE),
        "ensemble": {
            "model_a": {"id": MODEL_A_ID, "weight": MODEL_A_W},
            "model_b": {"id": MODEL_B_ID, "weight": MODEL_B_W},
        },
    })

@app.post("/detect")
def detect():
    t0 = time.time()

    language = None
    audio_format = None
    audio_bytes = None

    robust = False
    topk = DEFAULT_TOPK

    # 1) multipart/form-data
    if "file" in request.files:
        f = request.files["file"]
        audio_bytes = f.read()
        if f.filename and "." in f.filename:
            audio_format = f.filename.rsplit(".", 1)[-1].lower()
        language = request.form.get("language")

        robust = parse_bool(request.form.get("robust")) or (request.form.get("mode") == "robust")
        if request.form.get("topk") is not None:
            try:
                topk = int(request.form.get("topk"))
            except Exception:
                topk = DEFAULT_TOPK
    else:
        # 2) JSON base64
        payload = request.get_json(force=True, silent=True) or {}
        if "audio_b64" not in payload:
            return jsonify({"error": "Provide either multipart 'file' or JSON 'audio_b64'."}), 400

        language = payload.get("language")
        audio_format = payload.get("audio_format")

        robust = parse_bool(payload.get("robust")) or (payload.get("mode") == "robust")
        if payload.get("topk") is not None:
            try:
                topk = int(payload.get("topk"))
            except Exception:
                topk = DEFAULT_TOPK

        try:
            audio_bytes = base64.b64decode(payload["audio_b64"])
        except Exception as e:
            return jsonify({"error": f"Invalid base64: {str(e)}"}), 400

    # Decode
    try:
        wav, audio_meta = bytes_to_wav_16k_mono(audio_bytes, audio_format)
        wav_stats = {
            "min": float(wav.min().item()),
            "max": float(wav.max().item()),
            "mean": float(wav.mean().item()),
            "std": float(wav.std().item()),
        }
    except Exception as e:
        return jsonify({"error": f"Audio decode failed (check ffmpeg + format): {str(e)}"}), 400

    if audio_meta["duration_sec"] < 0.5:
        return jsonify({
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "explanation_text": "Audio too short (<0.5s) to make a reliable decision.",
            "explanation": {
                "reason": "Audio too short (<0.5s).",
                "language": language,
                "audio_meta": audio_meta,
                "waveform_stats": wav_stats,
            },
            "processing_ms": int((time.time() - t0) * 1000),
        }), 200

    analyzed_sec = FIXED_NUM_SAMPLES / TARGET_SR
    do_sliding = bool(robust and audio_meta["duration_sec"] > analyzed_sec + 0.1)

    p_bonafide = 0.0
    p_spoof = 0.0
    pad_action = "unchanged"
    mode = "robust" if do_sliding else "fast"
    window_stats: Optional[Dict[str, Any]] = None
    ensemble_debug: Optional[Dict[str, Any]] = None

    try:
        if not do_sliding:
            wav_fixed, pad_action = pad_or_truncate(wav, FIXED_NUM_SAMPLES, pad_mode="repeat")  # [1,64600]
            fused_probs, ensemble_debug = infer_probs_ensemble(wav_fixed)  # [1,2]
            p_bonafide = float(fused_probs[0, 0].item())
            p_spoof = float(fused_probs[0, 1].item())
        else:
            hop_len = int(HOP_SECONDS * TARGET_SR)
            win_len = FIXED_NUM_SAMPLES

            windows = get_windows(wav, win_len=win_len, hop_len=hop_len)  # [N,64600]
            fused_probs, ensemble_debug = infer_probs_ensemble(windows)    # [N,2]

            pb_list = fused_probs[:, 0].detach().cpu().tolist()
            ps_list = fused_probs[:, 1].detach().cpu().tolist()

            p_bonafide, p_spoof, stats = topk_mean_probs(pb_list, ps_list, k=topk)
            pad_action = "sliding_windows"

            window_stats = {
                **stats,
                "window_sec": float(win_len / TARGET_SR),
                "hop_sec": float(hop_len / TARGET_SR),
                "aggregation": f"topk_mean(k={max(1, min(int(topk), len(ps_list)))})",
            }
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    # Decision
    if p_spoof >= DEFAULT_THRESHOLD:
        classification = "AI_GENERATED"
        confidence = p_spoof
    else:
        classification = "HUMAN"
        confidence = p_bonafide

    explanation_text = make_explanation_text(
        classification=classification,
        confidence=confidence,
        p_bonafide=p_bonafide,
        p_spoof=p_spoof,
        threshold=DEFAULT_THRESHOLD,
        audio_meta=audio_meta,
        mode=mode,
        pad_action=pad_action,
        window_stats=window_stats,
    )

    return jsonify({
        "classification": classification,
        "confidence": round(float(confidence), 4),
        "explanation_text": explanation_text,
        "explanation": {
            "ensemble": {
                "model_a": {"id": MODEL_A_ID, "weight": MODEL_A_W},
                "model_b": {"id": MODEL_B_ID, "weight": MODEL_B_W},
                "debug": ensemble_debug,
            },
            "label_mapping": {"bonafide": "real(HUMAN)", "spoof": "fake(AI)"},
            "language": language,
            "audio_meta": audio_meta,
            "waveform_stats": wav_stats,
            "mode": mode,
            "preprocess": {
                "target_sample_rate": TARGET_SR,
                "fixed_num_samples": FIXED_NUM_SAMPLES,
                "pad_or_truncate": pad_action,
                "threshold_spoof": DEFAULT_THRESHOLD,
            },
            "windowing": window_stats,
            "scores": {
                "p_bonafide": round(float(p_bonafide), 6),
                "p_spoof": round(float(p_spoof), 6),
            },
        },
        "processing_ms": int((time.time() - t0) * 1000),
    }), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    # app.run(host="0.0.0.0", port=5000, debug=True)
