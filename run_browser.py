"""
run_browser.py

Unified local web app:

    audio file (drag & drop) --> NeMo ASR (NwachaMuna-NepConformer-Aug)
                              --> Newari autocorrect (correction_engine.py)
                              --> rendered side-by-side in the browser

Long clips are split into CHUNK_SECONDS-long segments before being fed to
the ASR model (one batched transcribe() call over all chunks), and the
per-chunk outputs are stitched back together before autocorrect runs on
the full text -- so everything downstream of raw_text/word_confidence is
unchanged.

Run:
    python run_browser.py

Then open:
    http://127.0.0.1:5000

Expects the three exported artifacts from the autocorrect notebook to be in
./artifacts/ :
    artifacts/dictionary.bin
    artifacts/symspell_index.bin
    artifacts/bigrams.bin
"""

import os
import time
import traceback
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfiltfilt, stft, istft
from math import gcd
from flask import Flask, request, jsonify, render_template_string

from correction_engine import CorrectionEngine

# NwachaMuna-NepConformer-Aug was trained at 16kHz mono (see the model's own
# training manifest config logged at startup: sample_rate: 16000). NeMo's
# transcribe() does NOT downmix/resample for you -- it feeds the file's raw
# shape straight into the model, so a stereo or non-16kHz upload crashes with
# "Input shape mismatch ... expected (batch, time) found (1, 2, N)" or silently
# degrades accuracy if it merely mismatches sample rate without erroring.
ASR_TARGET_SAMPLE_RATE = 16000

# Voice-enhancement preprocessing: this model was fine-tuned on relatively clean
# speech, so background noise (fans, traffic, room hum, mic hiss) measurably hurts
# it -- these three steps are the standard, dependency-free chain for cleaning
# speech before ASR (same family of technique as Audacity's Noise Reduction /
# the `noisereduce` library, just inlined via scipy so nothing extra to install).
# NOTE: this does NOT and CANNOT compensate for fast/rapid speech -- that's a
# fixed limitation of the model's own acoustic frame rate, not a signal-quality
# problem, so no amount of filtering here will fix a model mis-hearing someone
# talking quickly. It only helps with noisy/quiet/far-field recordings.
ASR_ENABLE_VOICE_ENHANCEMENT = True
ASR_HIGHPASS_CUTOFF_HZ = 80.0     # cuts rumble/handling noise/hum below typical voice fundamentals
ASR_NOISE_REDUCE_DB = 12.0        # how aggressively to gate out the estimated noise floor
ASR_TARGET_PEAK = 0.891           # ~-1 dBFS peak normalization target (headroom against clipping)
ASR_NORMALIZE_MAX_GAIN = 20.0      # never amplify more than this. Higher than a whole-file cap
                                    # would be safe, since this now only runs on VAD-confirmed
                                    # speech segments (see get_speech_segments), not raw audio
                                    # that might be pure background noise.
ASR_NORMALIZE_MIN_PEAK_FLOOR = 0.002  # a VAD-confirmed speech segment that's still this quiet
                                       # is likely a VAD false positive (silence/breath), not
                                       # worth normalizing -- skip rather than risk amplifying noise

# Long recordings are split at natural speech-vs-silence boundaries (VAD)
# instead of arbitrary fixed windows, so a split never lands mid-word or
# mid-sentence. CHUNK_SECONDS is now a SAFETY CAP only: if one continuous
# detected speech region is itself longer than this, it still gets sub-split
# (the model can't take unbounded input either way) -- but the normal case,
# a real sentence-length utterance, is never cut arbitrarily anymore.
CHUNK_SECONDS = 15

# Two speech regions separated by a pause shorter than this are merged into
# one segment, so a natural mid-sentence breath isn't mistaken for a sentence
# boundary and split into two (worse) transcriptions.
VAD_MERGE_GAP_SECONDS = 0.3
# Minimum length to keep a detected speech region; anything shorter is almost
# always a VAD false positive (click, breath, mouth noise) rather than speech.
VAD_MIN_SPEECH_SECONDS = 0.2

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ---------------------------------------------------------------------
# Lazy-loaded globals (ASR model is heavy, load once, on first request)
# ---------------------------------------------------------------------
_asr_model = None
_corrector = None
_vad_model = None
_vad_utils = None


def get_vad_model():
    """Lazily loads Silero VAD (via torch.hub) once per process. First call
    on a fresh machine needs internet access to download the model; it's
    cached under ~/.cache/torch/hub afterward, same as any other torch.hub
    model."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


def get_corrector():
    global _corrector
    if _corrector is None:
        # word_substitution_pairs.csv (produced by evaluate_asr_and_correction.py)
        # enables the confusion-correction pass if it's sitting in artifacts/ --
        # same "optional file, silent no-op if absent" pattern as the gazetteer.
        confusion_csv = os.path.join(ARTIFACTS_DIR, "word_substitution_pairs.csv")
        if os.path.exists(confusion_csv):
            print(f"[run_browser] Loading confusion pairs from: {confusion_csv}")
            _corrector = CorrectionEngine(ARTIFACTS_DIR, confusion_csv=confusion_csv)
        else:
            print(f"[run_browser] No {confusion_csv} found -- confusion-correction "
                  f"pass disabled (standard correction still applies). Copy your "
                  f"evaluation run's word_substitution_pairs.csv into artifacts/ to enable it.")
            _corrector = CorrectionEngine(ARTIFACTS_DIR)
    return _corrector


def get_asr_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict

        model = nemo_asr.models.ASRModel.from_pretrained(
            "ilprl-docse/NwachaMuna-NepConformer-Aug"
        )

        with open_dict(model.cfg.decoding):
            model.cfg.decoding.preserve_alignments = True
            model.cfg.decoding.compute_timestamps = True
            if "confidence_cfg" in model.cfg.decoding:
                model.cfg.decoding.confidence_cfg.preserve_token_confidence = True
                model.cfg.decoding.confidence_cfg.preserve_word_confidence = True

        model.change_decoding_strategy(model.cfg.decoding)
        _asr_model = model
    return _asr_model


# ---------------------------------------------------------------------
# Audio preprocessing (mono + 16kHz), required before every ASR call
# ---------------------------------------------------------------------

def _highpass_filter(data: np.ndarray, sr: int, cutoff_hz: float = ASR_HIGHPASS_CUTOFF_HZ,
                      order: int = 4) -> np.ndarray:
    """Removes energy below cutoff_hz (mic rumble, handling noise, AC hum,
    room tone) that sits below where voice fundamentals live -- cheap, safe,
    and never touches actual speech content."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, data).astype(np.float32)


def _spectral_denoise(data: np.ndarray, sr: int, reduce_db: float = ASR_NOISE_REDUCE_DB,
                       n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    """Spectral-gating noise reduction: estimates a noise magnitude profile from
    the quietest ~10% of frames in the clip (assumed background noise, since
    real speech frames are almost always louder than steady-state background
    noise), then subtracts that profile from every frame before reconstructing.
    This is the same underlying technique as the `noisereduce` library or
    Audacity's Noise Reduction, inlined here via scipy so there's no new
    dependency to install."""
    if len(data) < n_fft:
        return data  # too short to get a meaningful spectral estimate -- skip safely

    _, _, Zxx = stft(data, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    mag, phase = np.abs(Zxx), np.angle(Zxx)

    frame_energy = mag.mean(axis=0)
    n_noise_frames = max(1, int(0.1 * len(frame_energy)))
    quietest_idx = np.argsort(frame_energy)[:n_noise_frames]
    noise_profile = np.median(mag[:, quietest_idx], axis=1, keepdims=True)

    # Floor so gating never fully zeroes out a bin (that causes musical-noise
    # artifacts) -- always keep at least reduce_db worth of headroom below original.
    floor = mag * (10 ** (-reduce_db / 20))
    gated_mag = np.maximum(mag - noise_profile, floor)

    Zxx_denoised = gated_mag * np.exp(1j * phase)
    _, denoised = istft(Zxx_denoised, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # istft's output length can differ slightly from the input due to framing --
    # trim or zero-pad back to the original length so downstream code (VAD,
    # duration-based chunking) sees exactly what it expects.
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        denoised = np.pad(denoised, (0, len(data) - len(denoised)))
    return denoised.astype(np.float32)


def _normalize_peak(data: np.ndarray, target_peak: float = ASR_TARGET_PEAK,
                     max_gain: float = ASR_NORMALIZE_MAX_GAIN,
                     min_peak_floor: float = ASR_NORMALIZE_MIN_PEAK_FLOOR) -> np.ndarray:
    """Scales the whole clip so its loudest sample hits target_peak, without
    clipping. Quiet/far-field recordings are a common real-world failure mode
    for CTC models (low SNR relative to the model's training data), and this
    is the simplest lossless fix -- it doesn't change the *shape* of the
    signal, just its overall level.

    Two safety guards, because blind peak-normalization is dangerous on a clip
    that's mostly/entirely background noise (no real speech): if the required
    gain would exceed max_gain, or the clip is quieter than min_peak_floor to
    begin with, normalization is skipped -- otherwise a near-silent noise-only
    segment gets amplified into a loud false signal, which can both feed
    garbage to the ASR model and fool VAD into mistaking noise for speech."""
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak < min_peak_floor:
        return data.astype(np.float32)
    gain = target_peak / peak
    if gain > max_gain:
        return data.astype(np.float32)
    return (data * gain).astype(np.float32)


def enhance_audio_for_asr(data: np.ndarray, sr: int) -> np.ndarray:
    """Runs the whole-file part of the voice-enhancement chain: high-pass ->
    spectral denoise. Peak normalization deliberately is NOT done here --
    see _normalize_peak's docstring and get_speech_segments below for why
    it needs to happen per-VAD-confirmed-speech-segment instead of on the
    whole raw file. Wrapped so a pathological input (e.g. a near-silent or
    extremely short clip) degrades to the original signal rather than
    crashing the request -- enhancement is a quality improvement, not
    something that should ever turn a working transcription into a failed one."""
    if not ASR_ENABLE_VOICE_ENHANCEMENT:
        return data
    try:
        data = _highpass_filter(data, sr)
        data = _spectral_denoise(data, sr)
    except Exception:
        traceback.print_exc()
    return data


def preprocess_audio_for_asr(input_path: str, output_path: str,
                              target_sr: int = ASR_TARGET_SAMPLE_RATE) -> None:
    """Reads any audio soundfile can open, downmixes to mono if needed,
    resamples to target_sr if needed, and writes a clean 16-bit PCM wav to
    output_path. This is what actually fixes the
    'Input shape mismatch ... expected (batch, time)' crash: NeMo's
    transcribe() expects a plain (time,) mono signal at the model's trained
    sample rate, and does not do this conversion itself.

    Also runs the noise/normalization enhancement chain (see
    enhance_audio_for_asr) before writing, since VAD and ASR both do better
    on a cleaned signal than the raw upload."""
    data, sr = sf.read(input_path, always_2d=True)  # shape: (frames, channels)

    # Downmix to mono by averaging channels (stereo/multi-channel -> mono).
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to the model's expected sample rate if it doesn't already match.
    if sr != target_sr:
        g = gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        data = resample_poly(data, up, down)

    data = data.astype(np.float32)
    data = enhance_audio_for_asr(data, target_sr)
    sf.write(output_path, data, target_sr, subtype="PCM_16")


# ---------------------------------------------------------------------
# Segmentation for long audio: use VAD to find actual speech regions
# (silence/pauses excluded) instead of cutting at arbitrary fixed-length
# boundaries, so a split never lands mid-word or mid-sentence. Any single
# detected speech region still longer than CHUNK_SECONDS is sub-split as a
# safety net -- the model can't take unbounded input either way -- but that
# only kicks in for unusually long continuous speech, not the normal case.
# ---------------------------------------------------------------------

def get_speech_segments(processed_path: str, chunk_dir: str, tag: str,
                         chunk_seconds: int = CHUNK_SECONDS,
                         sr: int = ASR_TARGET_SAMPLE_RATE):
    """Splits an already-preprocessed (mono, sr Hz) wav at speech/silence
    boundaries detected by Silero VAD, so each resulting file corresponds to
    one actual spoken sentence/utterance rather than an arbitrary time slice.

    Returns a list of dicts, one per written wav file, in chronological
    order:
        {"path": <wav file path>, "group_id": <int>,
         "start_sec": <float>, "end_sec": <float>}

    `group_id` ties pieces back together: normally one VAD speech region ==
    one group == one output file, but a region longer than chunk_seconds is
    sub-split into multiple pieces that share the same group_id, so the
    caller can re-join them into a single sentence's transcript afterward.

    If VAD can't be loaded (e.g. no internet on first run) or finds no
    speech at all, falls back to treating the whole file as one segment --
    same behavior as the old single-file passthrough case.
    """
    data, file_sr = sf.read(processed_path, always_2d=False)
    if file_sr != sr:
        raise ValueError(f"get_speech_segments expects {sr}Hz audio, got {file_sr}Hz")
    data = data.astype(np.float32)
    total_samples = len(data)
    chunk_samples = int(chunk_seconds * sr)
    merge_gap_samples = int(VAD_MERGE_GAP_SECONDS * sr)
    min_speech_samples = int(VAD_MIN_SPEECH_SECONDS * sr)

    def whole_file_fallback():
        return [{"path": processed_path, "group_id": 0,
                  "start_sec": 0.0, "end_sec": total_samples / sr}]

    try:
        import torch
        vad_model, vad_utils = get_vad_model()
        get_speech_timestamps = vad_utils[0]
        speech_ts = get_speech_timestamps(
            torch.from_numpy(data), vad_model, sampling_rate=sr,
        )  # list of {"start": sample_idx, "end": sample_idx}, chronological
    except Exception:
        # VAD unavailable for any reason -- degrade gracefully instead of
        # failing the whole request; the old fixed-window/no-split path
        # still guarantees correct (if less precise) behavior.
        speech_ts = None

    if not speech_ts:
        if total_samples <= chunk_samples:
            return whole_file_fallback()
        # No VAD, but the file is long -- fall back to the old fixed-window
        # behavior rather than feeding the model an overlong single clip.
        speech_ts = [{"start": s, "end": min(s + chunk_samples, total_samples)}
                     for s in range(0, total_samples, chunk_samples)]

    # Merge regions separated by a short pause -- avoids splitting one
    # sentence into two just because of a natural breath/pause.
    merged = []
    for seg in speech_ts:
        if merged and seg["start"] - merged[-1]["end"] <= merge_gap_samples:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append({"start": seg["start"], "end": seg["end"]})

    # Drop VAD false positives that are implausibly short to be real speech.
    merged = [seg for seg in merged if seg["end"] - seg["start"] >= min_speech_samples]
    if not merged:
        return whole_file_fallback()

    segments = []
    group_id = 0
    for seg in merged:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_len = seg_end - seg_start
        if seg_len <= chunk_samples:
            piece_bounds = [(seg_start, seg_end)]
        else:
            # Safety-net sub-split: this region is unusually long for one
            # continuous utterance, so cap each piece at chunk_seconds.
            piece_bounds = [
                (s, min(s + chunk_samples, seg_end))
                for s in range(seg_start, seg_end, chunk_samples)
            ]
        for i, (p_start, p_end) in enumerate(piece_bounds):
            piece_data = data[p_start:p_end]
            # Safe to peak-normalize here specifically because this piece already
            # passed VAD + the min-speech-length filter above -- it's confirmed to
            # contain real speech, not just background noise, unlike normalizing
            # the raw whole-file upload would be (see _normalize_peak's docstring).
            if ASR_ENABLE_VOICE_ENHANCEMENT:
                piece_data = _normalize_peak(piece_data)
            piece_path = os.path.join(chunk_dir, f"seg_{tag}_{group_id}_{i}.wav")
            sf.write(piece_path, piece_data, sr, subtype="PCM_16")
            segments.append({
                "path": piece_path,
                "group_id": group_id,
                "start_sec": p_start / sr,
                "end_sec": p_end / sr,
            })
        group_id += 1

    return segments


# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use wav/flac/mp3/ogg/m4a."}), 400

    ts = int(time.time() * 1000)
    safe_name = f"upload_{ts}{ext}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(path)

    processed_path = os.path.join(UPLOAD_DIR, f"processed_{safe_name}.wav")
    segment_paths = []
    try:
        preprocess_audio_for_asr(path, processed_path)
    except Exception as e:
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"error": f"Could not read/convert audio file: {e}"}), 400

    try:
        model = get_asr_model()
        corrector = get_corrector()

        # Split at speech/silence boundaries (VAD) so each piece is one real
        # spoken sentence/utterance rather than an arbitrary time slice.
        # A short clip that's a single sentence yields exactly one segment.
        segments = get_speech_segments(processed_path, UPLOAD_DIR, tag=str(ts))
        segment_paths = [seg["path"] for seg in segments]

        hypotheses = model.transcribe(segment_paths, return_hypotheses=True)

        # Regroup pieces back into sentences via group_id (multiple pieces
        # share a group_id only when one long VAD speech region had to be
        # safety-net sub-split at CHUNK_SECONDS).
        sentences = []
        all_word_conf_pairs = []
        cur_group_id = None
        cur_texts, cur_confs = [], []
        cur_start = cur_end = None

        def flush_sentence():
            if not cur_texts:
                return
            text = " ".join(t for t in cur_texts if t).strip()
            if not text:
                return
            corrected, changes = corrector.correct_text(text)
            sentence_words = text.split()
            # Same rule as the original single-blob logic: only trust
            # per-word confidence if the count actually lines up with the
            # word count -- otherwise leave confidence unset for this
            # sentence's words rather than mis-align them.
            if cur_confs and len(cur_confs) == len(sentence_words):
                for w, c in zip(sentence_words, cur_confs):
                    all_word_conf_pairs.append({"word": w, "confidence": float(c)})
            else:
                for w in sentence_words:
                    all_word_conf_pairs.append({"word": w, "confidence": None})
            sentences.append({
                "start": round(cur_start, 2),
                "end": round(cur_end, 2),
                "text": text,
                "corrected_text": corrected,
                "changes": changes,
                "avg_confidence": (
                    float(sum(cur_confs) / len(cur_confs)) if cur_confs else None
                ),
            })

        for seg, hyp in zip(segments, hypotheses):
            if seg["group_id"] != cur_group_id:
                flush_sentence()
                cur_group_id = seg["group_id"]
                cur_texts, cur_confs = [], []
                cur_start = seg["start_sec"]
            cur_texts.append((hyp.text or "").strip())
            cur_end = seg["end_sec"]
            wc = getattr(hyp, "word_confidence", None)
            if wc:
                cur_confs.extend(wc)
        flush_sentence()

        # Backward-compatible flat fields, now newline-joined per sentence
        # instead of one run-on paragraph.
        raw_text = "\n".join(s["text"] for s in sentences)
        corrected_text = "\n".join(s["corrected_text"] for s in sentences)
        changes = [c for s in sentences for c in s["changes"]]

        all_confs = [s["avg_confidence"] for s in sentences if s["avg_confidence"] is not None]
        avg_confidence = float(sum(all_confs) / len(all_confs)) if all_confs else None

        word_conf_pairs = all_word_conf_pairs

        return jsonify({
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "changes": changes,
            "word_confidence": word_conf_pairs,
            "avg_confidence": avg_confidence,
            "sentences": sentences,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        cleanup_paths = [path, processed_path]
        # Only remove segment files that are separate from processed_path
        # (the whole-file VAD fallback reuses processed_path directly, so
        # it's already in the cleanup list above and shouldn't be double-listed).
        cleanup_paths += [p for p in segment_paths if p != processed_path]
        for p in cleanup_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------
# Front end (single-file template: manuscript / palm-leaf theme)
# ---------------------------------------------------------------------

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Newari speech transcription</title>
<style>
  :root {
    --colour-text: #0b0c0c;
    --colour-text-secondary: #505a5f;
    --colour-link: #1d70b8;
    --colour-link-hover: #003078;
    --colour-border: #b1b4b6;
    --colour-border-light: #f3f2f1;
    --colour-background: #ffffff;
    --colour-background-alt: #f3f2f1;
    --colour-focus: #ffdd00;
    --colour-button: #00703c;
    --colour-button-hover: #005a30;
    --colour-error: #d4351c;
    --colour-highlight: #fff7bf;
    --font-stack: -apple-system, "Segoe UI", Arial, sans-serif;
    --max-width: 760px;
  }

  * { box-sizing: border-box; }

  html { font-size: 16px; }

  body {
    margin: 0;
    font-family: var(--font-stack);
    color: var(--colour-text);
    background: var(--colour-background);
    line-height: 1.5;
    font-size: 1.1875rem;
  }

  a { color: var(--colour-link); }
  a:hover { color: var(--colour-link-hover); }
  a:focus, button:focus, input:focus, .dropzone:focus {
    outline: 3px solid var(--colour-focus);
    outline-offset: 0;
    box-shadow: inset 0 0 0 2px var(--colour-text);
  }

  .skip-link {
    position: absolute;
    left: -9999px;
    top: 0;
    background: #fff;
    padding: 0.5rem 1rem;
    z-index: 100;
  }
  .skip-link:focus { left: 0; }

  header.service-header {
    background: var(--colour-text);
    color: #fff;
    padding: 0.75rem 1.25rem;
  }
  header.service-header .inner {
    max-width: var(--max-width);
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  header.service-header .logo-mark {
    flex-shrink: 0;
    display: block;
  }
  header.service-header .org {
    font-weight: 700;
    font-size: 1.125rem;
    letter-spacing: 0.02em;
  }
  header.service-header .service-name {
    font-size: 1rem;
    color: #cfd3d6;
    text-decoration: none;
    border-left: 1px solid #6f777b;
    padding-left: 0.75rem;
  }
  /* Waveform logo bars animate gently -- a small, honest bit of life on an
     otherwise plain page, standing in for "listening" without resorting to
     a stock AI-sparkle icon. Respects reduced-motion preference below. */
  .logo-mark .bar {
    animation: logo-pulse 1.6s ease-in-out infinite;
    transform-origin: center;
  }
  .logo-mark .bar:nth-child(1) { animation-delay: 0s; }
  .logo-mark .bar:nth-child(2) { animation-delay: 0.15s; }
  .logo-mark .bar:nth-child(3) { animation-delay: 0.3s; }
  .logo-mark .bar:nth-child(4) { animation-delay: 0.15s; }
  .logo-mark .bar:nth-child(5) { animation-delay: 0s; }
  @keyframes logo-pulse {
    0%, 100% { transform: scaleY(0.4); }
    50% { transform: scaleY(1); }
  }
  @media (prefers-reduced-motion: reduce) {
    .logo-mark .bar { animation: none; transform: scaleY(0.75); }
  }

  main {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
  }

  h1 {
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 0.75rem;
  }

  p.lede {
    font-size: 1.1875rem;
    color: var(--colour-text-secondary);
    max-width: 42em;
    margin: 0 0 2rem;
  }

  .panel {
    border: 1px solid var(--colour-border);
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  label.field-label {
    display: block;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }

  .dropzone {
    border: 2px dashed var(--colour-border);
    background: var(--colour-background-alt);
    padding: 2rem 1rem;
    text-align: center;
    cursor: pointer;
  }
  .dropzone.dragover {
    border-color: var(--colour-link);
    background: #eef4f8;
  }
  .dropzone p { margin: 0.25rem 0; }
  .dropzone .hint { color: var(--colour-text-secondary); font-size: 0.9375rem; }
  .dropzone .filename {
    margin-top: 0.75rem;
    font-weight: 700;
    word-break: break-all;
  }
  input[type="file"] {
    position: absolute;
    width: 1px; height: 1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
  }

  button.primary {
    font-family: inherit;
    font-size: 1.1875rem;
    font-weight: 700;
    background: var(--colour-button);
    color: #fff;
    border: 2px solid transparent;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    box-shadow: 0 2px 0 #002d18;
  }
  button.primary:hover { background: var(--colour-button-hover); }
  button.primary:active { box-shadow: none; transform: translateY(2px); }
  button.primary:disabled {
    background: #b1b4b6;
    box-shadow: none;
    cursor: not-allowed;
  }
  button.primary .arrow { margin-left: 0.4rem; }

  .status {
    margin-top: 1rem;
    font-weight: 700;
  }
  .status[data-state="error"] { color: var(--colour-error); }

  .error-summary {
    border: 4px solid var(--colour-error);
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
  }
  .error-summary h2 {
    color: var(--colour-error);
    font-size: 1.25rem;
    margin: 0 0 0.5rem;
  }

  h2.section-heading {
    font-size: 1.5rem;
    border-bottom: 1px solid var(--colour-border);
    padding-bottom: 0.5rem;
    margin-top: 0;
  }

  table.sentence-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1.5rem;
    font-size: 1rem;
  }
  table.sentence-table caption {
    text-align: left;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  table.sentence-table th,
  table.sentence-table td {
    text-align: left;
    vertical-align: top;
    padding: 0.75rem 0.75rem 0.75rem 0;
    border-bottom: 1px solid var(--colour-border-light);
  }
  table.sentence-table th {
    border-bottom: 2px solid var(--colour-border);
    font-size: 0.9375rem;
    color: var(--colour-text-secondary);
    font-weight: 700;
  }
  table.sentence-table td.timestamp {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    color: var(--colour-text-secondary);
    width: 4.5rem;
  }
  table.sentence-table td.confidence {
    white-space: nowrap;
    width: 4rem;
  }
  .changed-word {
    background: var(--colour-highlight);
    padding: 0 0.15em;
  }
  .conf-low { color: var(--colour-error); font-weight: 700; }

  details.changes-detail {
    margin-top: 1rem;
  }
  details.changes-detail summary {
    cursor: pointer;
    font-weight: 700;
    color: var(--colour-link);
  }
  details.changes-detail summary:hover { text-decoration: underline; }
  ul.change-list {
    margin: 0.75rem 0 0;
    padding-left: 1.25rem;
  }
  ul.change-list li { margin-bottom: 0.35rem; }

  .visually-hidden {
    position: absolute;
    width: 1px; height: 1px;
    margin: -1px; padding: 0; border: 0;
    clip: rect(0 0 0 0);
    overflow: hidden;
  }

  footer.site-footer {
    border-top: 1px solid var(--colour-border);
    padding: 2rem 1.25rem;
    margin-top: 3rem;
  }
  footer.site-footer .inner {
    max-width: var(--max-width);
    margin: 0 auto;
    color: var(--colour-text-secondary);
    font-size: 0.9375rem;
  }

  @media (max-width: 480px) {
    main { padding: 1.5rem 1rem 3rem; }
    h1 { font-size: 1.5rem; }
    .panel { padding: 1rem; }
  }
</style>
</head>
<body>

<a class="skip-link" href="#main-content">Skip to main content</a>

<header class="service-header">
  <div class="inner">
    <span class="org">Newari Speech Tools</span>
    <span class="service-name">Transcription and autocorrect</span>
  </div>
</header>

<main id="main-content">
  <h1>Transcribe a Newari audio recording</h1>
  <p class="lede">
    Upload a recording and this tool will produce a transcript, then apply
    automatic spelling correction. Accepted formats: WAV, FLAC, MP3, OGG, M4A.
  </p>

  <div class="panel">
    <label class="field-label" for="audio-input">Audio file</label>
    <div class="dropzone" id="dropzone" tabindex="0" role="button"
         aria-describedby="dropzone-hint">
      <p>Drag and drop a file here, or select a file</p>
      <p class="hint" id="dropzone-hint">Maximum file size 200MB.</p>
      <p class="filename" id="filename" aria-live="polite"></p>
    </div>
    <input type="file" id="audio-input" accept=".wav,.flac,.mp3,.ogg,.m4a">

    <p style="margin-top:1.5rem;">
      <button class="primary" id="transcribe-btn" type="button" disabled>
        Transcribe recording <span class="arrow" aria-hidden="true">&rarr;</span>
      </button>
    </p>

    <p class="status" id="status" role="status" aria-live="polite"></p>
  </div>

  <div id="error-container"></div>
  <div id="results-container"></div>
</main>

<footer class="site-footer">
  <div class="inner">
    <p>Audio is processed locally and deleted immediately after transcription.</p>
  </div>
</footer>

<script>
(function () {
  var dropzone = document.getElementById('dropzone');
  var fileInput = document.getElementById('audio-input');
  var filenameEl = document.getElementById('filename');
  var transcribeBtn = document.getElementById('transcribe-btn');
  var statusEl = document.getElementById('status');
  var errorContainer = document.getElementById('error-container');
  var resultsContainer = document.getElementById('results-container');
  var selectedFile = null;

  function setFile(file) {
    selectedFile = file;
    filenameEl.textContent = file ? file.name : '';
    transcribeBtn.disabled = !file;
  }

  dropzone.addEventListener('click', function () { fileInput.click(); });
  dropzone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  dropzone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', function () {
    dropzone.classList.remove('dragover');
  });
  dropzone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', function () {
    if (fileInput.files && fileInput.files.length) {
      setFile(fileInput.files[0]);
    }
  });

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatTimestamp(seconds) {
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function renderCorrectedWithHighlights(correctedText, changes) {
    if (!changes || !changes.length) { return escapeHtml(correctedText); }
    var correctedSet = {};
    changes.forEach(function (c) { correctedSet[c.corrected] = true; });
    var words = correctedText.split(/(\s+)/);
    return words.map(function (w) {
      var trimmed = w.trim();
      if (trimmed && correctedSet[trimmed]) {
        return '<span class="changed-word">' + escapeHtml(w) + '</span>';
      }
      return escapeHtml(w);
    }).join('');
  }

  function renderResults(data) {
    resultsContainer.innerHTML = '';
    errorContainer.innerHTML = '';

    if (!data.sentences || !data.sentences.length) {
      resultsContainer.innerHTML =
        '<p>No speech was detected in this recording.</p>';
      return;
    }

    var html = '<h2 class="section-heading">Transcript</h2>';
    html += '<table class="sentence-table">';
    html += '<caption class="visually-hidden">Transcribed sentences with corrections and confidence</caption>';
    html += '<thead><tr>' +
      '<th scope="col">Time</th>' +
      '<th scope="col">As heard</th>' +
      '<th scope="col">Corrected</th>' +
      '<th scope="col">Confidence</th>' +
      '</tr></thead><tbody>';

    var allChanges = [];

    data.sentences.forEach(function (s) {
      var confPct = (s.avg_confidence !== null && s.avg_confidence !== undefined)
        ? Math.round(s.avg_confidence * 100) + '%' : '—';
      var confClass = (s.avg_confidence !== null && s.avg_confidence !== undefined && s.avg_confidence < 0.6)
        ? ' class="confidence conf-low"' : ' class="confidence"';

      html += '<tr>';
      html += '<td class="timestamp">' + formatTimestamp(s.start) + '</td>';
      html += '<td>' + escapeHtml(s.text) + '</td>';
      html += '<td>' + renderCorrectedWithHighlights(s.corrected_text, s.changes) + '</td>';
      html += '<td' + confClass + '>' + confPct + '</td>';
      html += '</tr>';

      if (s.changes && s.changes.length) {
        allChanges = allChanges.concat(s.changes);
      }
    });

    html += '</tbody></table>';

    if (allChanges.length) {
      html += '<details class="changes-detail">';
      html += '<summary>' + allChanges.length + ' correction' +
        (allChanges.length === 1 ? '' : 's') + ' applied</summary>';
      html += '<ul class="change-list">';
      allChanges.forEach(function (c) {
        html += '<li>' + escapeHtml(c.original) + ' &rarr; ' + escapeHtml(c.corrected) + '</li>';
      });
      html += '</ul></details>';
    }

    resultsContainer.innerHTML = html;
  }

  function renderError(message) {
    errorContainer.innerHTML =
      '<div class="error-summary" role="alert">' +
      '<h2>There is a problem</h2>' +
      '<p>' + escapeHtml(message) + '</p>' +
      '</div>';
    resultsContainer.innerHTML = '';
  }

  transcribeBtn.addEventListener('click', function () {
    if (!selectedFile) { return; }

    transcribeBtn.disabled = true;
    statusEl.removeAttribute('data-state');
    statusEl.textContent = 'Transcribing recording. This may take a moment.';
    errorContainer.innerHTML = '';
    resultsContainer.innerHTML = '';

    var formData = new FormData();
    formData.append('audio', selectedFile);

    fetch('/api/transcribe', { method: 'POST', body: formData })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, data: data };
        });
      })
      .then(function (result) {
        transcribeBtn.disabled = false;
        if (!result.ok || result.data.error) {
          statusEl.setAttribute('data-state', 'error');
          statusEl.textContent = 'Transcription failed.';
          renderError(result.data.error || 'An unknown error occurred.');
          return;
        }
        statusEl.textContent = 'Transcription complete.';
        renderResults(result.data);
      })
      .catch(function (err) {
        transcribeBtn.disabled = false;
        statusEl.setAttribute('data-state', 'error');
        statusEl.textContent = 'Transcription failed.';
        renderError('Could not reach the server: ' + err.message);
      });
  });
})();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    print("Starting on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)