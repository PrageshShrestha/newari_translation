"""
run_browser.py

Unified local web app:

    audio file (drag & drop, OR recorded live in the browser mic)
                              --> NeMo ASR (NwachaMuna-NepConformer-Aug)
                              --> Newari autocorrect (correction_engine.py)
                              --> rendered side-by-side in the browser

Long clips are split into CHUNK_SECONDS-long segments before being fed to
the ASR model (one batched transcribe() call over all chunks), and the
per-chunk outputs are stitched back together before autocorrect runs on
the full text -- so everything downstream of raw_text/word_confidence is
unchanged.

Both upload paths (drag-and-drop file, and in-browser microphone recording)
converge on the exact same /api/transcribe endpoint and the exact same
preprocess_audio_for_asr() -> get_speech_segments() -> transcribe() ->
correct_text() pipeline below -- a mic recording is just another audio
file as far as the backend is concerned, it only differs in how the
browser produced it (see MIME/container note next to ALLOWED_EXT).

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
import shutil
import subprocess
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

# .webm and .ogg are included specifically for in-browser microphone
# recordings: MediaRecorder in Chrome/Firefox/Edge produces audio/webm
# (Opus) by default, and Safari/some mobile browsers fall back to
# audio/mp4 or audio/ogg depending on what codecs are available -- so all
# of these need to be accepted uploads, not just the "file picker" formats.
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm", ".mp4"}

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


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _convert_with_ffmpeg(input_path: str, target_sr: int) -> str:
    """Decodes any container/codec ffmpeg understands (webm/Opus and
    mp4/AAC in particular -- the formats produced by browser MediaRecorder,
    which libsndfile/soundfile cannot read directly) into a temporary mono
    WAV at target_sr. Raises if ffmpeg is missing or the conversion fails,
    so the caller can surface a clear error instead of a confusing
    downstream soundfile crash."""
    if not _ffmpeg_available():
        raise RuntimeError(
            "This audio format needs ffmpeg to decode (e.g. a browser "
            "microphone recording) but ffmpeg is not installed on the "
            "server. Install ffmpeg, or upload a .wav/.flac file instead."
        )
    tmp_path = input_path + ".decoded.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", str(target_sr),
            tmp_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(tmp_path):
        stderr_tail = result.stderr.decode("utf-8", errors="ignore")[-500:]
        raise RuntimeError(f"ffmpeg could not decode this audio file: {stderr_tail}")
    return tmp_path


def preprocess_audio_for_asr(input_path: str, output_path: str,
                              target_sr: int = ASR_TARGET_SAMPLE_RATE) -> None:
    """Reads any audio soundfile can open, downmixes to mono if needed,
    resamples to target_sr if needed, and writes a clean 16-bit PCM wav to
    output_path. This is what actually fixes the
    'Input shape mismatch ... expected (batch, time)' crash: NeMo's
    transcribe() expects a plain (time,) mono signal at the model's trained
    sample rate, and does not do this conversion itself.

    Browser microphone recordings arrive as webm/Opus (or occasionally
    mp4/AAC on Safari) rather than a libsndfile-readable container, so
    sf.read() is tried first for the common file-upload case and, if that
    fails, falls back to an ffmpeg decode pass -- this is what lets a mic
    recording go through the exact same function as a dropped-in file.

    Also runs the noise/normalization enhancement chain (see
    enhance_audio_for_asr) before writing, since VAD and ASR both do better
    on a cleaned signal than the raw upload."""
    decoded_tmp_path = None
    try:
        data, sr = sf.read(input_path, always_2d=True)  # shape: (frames, channels)
    except Exception:
        # Not something libsndfile can open directly -- most commonly a
        # browser mic recording (webm/Opus, mp4/AAC). Decode via ffmpeg and
        # retry the read against the decoded wav.
        decoded_tmp_path = _convert_with_ffmpeg(input_path, target_sr)
        data, sr = sf.read(decoded_tmp_path, always_2d=True)

    try:
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
    finally:
        if decoded_tmp_path is not None:
            try:
                os.remove(decoded_tmp_path)
            except OSError:
                pass


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
        return jsonify({"error": f"Unsupported file type '{ext}'. Use wav/flac/mp3/ogg/m4a/webm."}), 400

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
<title>Shruti — Nepal Bhasa Speech Recognition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Noto+Serif+Devanagari:wght@500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --brick:        #8B2E1D;
    --brick-deep:    #6E2216;
    --copper:       #B5651D;
    --manuscript:   #F4E7D3;
    --manuscript-2: #EADBC2;
    --wood:         #3A2418;
    --wood-soft:    #5A3A28;
    --gold:         #D4A017;
    --gold-soft:    #E4BE55;
    --ink:          #2A1810;
    --paper-line:   rgba(58, 36, 24, 0.10);
    --shadow-warm:  rgba(58, 24, 12, 0.25);
    --record-red:   #B5301D;

    --font-display: "Fraunces", "Noto Serif Devanagari", serif;
    --font-body:    "Fraunces", "Noto Serif Devanagari", serif;
    --font-mono:    "JetBrains Mono", ui-monospace, monospace;

    --max-width: 880px;
  }

  * { box-sizing: border-box; }
  html { font-size: 16px; }

  body {
    margin: 0;
    font-family: var(--font-body);
    color: var(--ink);
    line-height: 1.6;
    font-size: 1.0625rem;
    background:
      radial-gradient(1100px 620px at 12% -8%, rgba(212,160,23,0.16), transparent 60%),
      radial-gradient(900px 560px at 108% 12%, rgba(139,46,29,0.14), transparent 55%),
      repeating-linear-gradient(135deg, rgba(58,36,24,0.035) 0 2px, transparent 2px 26px),
      repeating-linear-gradient(45deg, rgba(58,36,24,0.03) 0 2px, transparent 2px 26px),
      var(--manuscript);
  }

  a { color: var(--brick); }
  a:hover { color: var(--brick-deep); }
  a:focus-visible, button:focus-visible, input:focus-visible, .dropzone:focus-visible {
    outline: 3px solid var(--gold);
    outline-offset: 2px;
  }

  .skip-link {
    position: absolute; left: -9999px; top: 0;
    background: var(--wood); color: var(--manuscript);
    padding: 0.6rem 1rem; z-index: 100;
    font-family: var(--font-mono); font-size: 0.85rem;
  }
  .skip-link:focus { left: 0; }

  /* ---------------- Header ---------------- */
  header.top-bar {
    background: linear-gradient(180deg, var(--wood) 0%, #2c1a10 100%);
    color: var(--manuscript);
    border-bottom: 3px solid var(--gold);
  }
  header.top-bar .inner {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 0.9rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
  }
  header.top-bar .wordmark {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.2rem;
    letter-spacing: 0.02em;
    color: var(--manuscript);
  }
  header.top-bar .tagline-sm {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--gold-soft);
    letter-spacing: 0.05em;
    border-left: 1px solid rgba(244,231,211,0.25);
    padding-left: 0.85rem;
  }

  /* ---------------- Logo mark ----------------
     Stepped pagoda-roof silhouette above a row of bars that read equally
     as a waveform and as the vertical struts of a carved Newari jhya
     (lattice window). Bars breathe gently -- a small sign of "listening"
     rather than a static icon. */
  .logo-mark { flex-shrink: 0; display: block; }
  .logo-mark .roof { fill: var(--gold); }
  .logo-mark .bar {
    fill: var(--copper);
    animation: logo-breathe 2.2s ease-in-out infinite;
    transform-origin: center bottom;
  }
  .logo-mark .bar:nth-child(3) { animation-delay: 0s;    fill: var(--gold); }
  .logo-mark .bar:nth-child(4) { animation-delay: 0.18s; }
  .logo-mark .bar:nth-child(5) { animation-delay: 0.36s; }
  .logo-mark .bar:nth-child(6) { animation-delay: 0.18s; }
  .logo-mark .bar:nth-child(7) { animation-delay: 0s; }
  @keyframes logo-breathe {
    0%, 100% { transform: scaleY(0.55); }
    50%      { transform: scaleY(1); }
  }
  @media (prefers-reduced-motion: reduce) {
    .logo-mark .bar { animation: none; transform: scaleY(0.8); }
  }

  /* ---------------- Hero ---------------- */
  .hero {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 3.5rem 1.25rem 2.5rem;
    text-align: center;
  }
  .hero .devanagari {
    font-family: "Noto Serif Devanagari", serif;
    font-weight: 700;
    font-size: clamp(2.4rem, 7vw, 3.6rem);
    color: var(--brick);
    margin: 0 0 0.35rem;
    letter-spacing: 0.01em;
  }
  .hero h1 {
    font-family: var(--font-display);
    font-weight: 600;
    font-style: italic;
    font-size: clamp(1.15rem, 3vw, 1.5rem);
    color: var(--wood-soft);
    margin: 0 0 1.1rem;
    letter-spacing: 0.01em;
  }
  .hero p.lede {
    max-width: 42em;
    margin: 0 auto;
    color: var(--wood-soft);
    font-size: 1.05rem;
  }
  .hero .rule {
    width: 72px; height: 3px;
    margin: 1.6rem auto 0;
    background: linear-gradient(90deg, var(--copper), var(--gold));
    border-radius: 2px;
  }

  main { max-width: var(--max-width); margin: 0 auto; padding: 0 1.25rem 4rem; }

  section { margin-bottom: 3rem; }

  h2.section-heading {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--brick);
    margin: 0 0 1.1rem;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
  }
  h2.section-heading .num {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--gold);
    font-weight: 600;
  }

  /* ---------------- Panel (carved-frame card) ---------------- */
  .panel {
    background: #FBF3E4;
    border: 1px solid var(--paper-line);
    border-radius: 10px;
    padding: 1.75rem;
    box-shadow: 0 10px 28px -18px var(--shadow-warm), inset 0 0 0 1px rgba(212,160,23,0.15);
    position: relative;
  }
  .panel::before {
    /* thin gold inlay line, evoking a carved wooden frame */
    content: "";
    position: absolute; inset: 6px;
    border: 1px solid rgba(212,160,23,0.25);
    border-radius: 6px;
    pointer-events: none;
  }

  label.field-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--copper);
    font-weight: 600;
    margin-bottom: 0.75rem;
  }

  /* ---------------- Source tabs (Upload file / Record audio) ---------------- */
  .source-tabs {
    display: inline-flex;
    gap: 0.35rem;
    padding: 0.3rem;
    background: var(--manuscript-2);
    border-radius: 8px;
    margin-bottom: 1.25rem;
  }
  .source-tabs button {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: transparent;
    border: none;
    color: var(--wood-soft);
    padding: 0.55rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    transition: background .15s ease, color .15s ease;
  }
  .source-tabs button svg { flex-shrink: 0; }
  .source-tabs button:hover { color: var(--wood); }
  .source-tabs button.active {
    background: var(--wood);
    color: var(--gold-soft);
  }
  .source-panel { display: none; }
  .source-panel.active { display: block; }

  .dropzone {
    border: 2px dashed var(--copper);
    background: repeating-linear-gradient(45deg, rgba(181,101,29,0.05) 0 10px, transparent 10px 20px);
    border-radius: 8px;
    padding: 2.5rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s ease, background .2s ease;
  }
  .dropzone:hover { border-color: var(--gold); }
  .dropzone.dragover { border-color: var(--gold); background: rgba(212,160,23,0.10); }
  .dropzone p { margin: 0.3rem 0; }
  .dropzone .glyph {
    font-family: "Noto Serif Devanagari", serif;
    font-size: 1.7rem;
    color: var(--gold);
    display: block;
    margin-bottom: 0.4rem;
  }
  .dropzone .instruction { font-size: 1.05rem; color: var(--wood); }
  .dropzone .hint { color: var(--wood-soft); font-size: 0.85rem; font-family: var(--font-mono); }
  .dropzone .filename {
    margin-top: 0.85rem;
    font-weight: 600;
    word-break: break-all;
    color: var(--brick);
  }
  input[type="file"] {
    position: absolute; width: 1px; height: 1px;
    overflow: hidden; clip: rect(0 0 0 0);
  }

  /* ---------------- Recorder panel ---------------- */
  .recorder {
    border: 2px dashed var(--copper);
    background: repeating-linear-gradient(45deg, rgba(181,101,29,0.05) 0 10px, transparent 10px 20px);
    border-radius: 8px;
    padding: 2.25rem 1rem;
    text-align: center;
  }
  .recorder.is-recording {
    border-color: var(--record-red);
    background: rgba(181,48,29,0.06);
  }
  .record-btn {
    width: 76px; height: 76px;
    border-radius: 50%;
    border: none;
    background: linear-gradient(180deg, var(--record-red), #8a2314);
    color: var(--manuscript);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 3px 0 #641a0f, 0 10px 20px -8px var(--shadow-warm);
    transition: transform .12s ease, box-shadow .12s ease;
    position: relative;
  }
  .record-btn:hover { transform: translateY(-1px); }
  .record-btn:active { box-shadow: 0 1px 0 #641a0f; transform: translateY(2px); }
  .record-btn .mic-icon { width: 26px; height: 26px; }
  .record-btn .stop-icon { width: 22px; height: 22px; display: none; }
  .record-btn.recording {
    animation: record-pulse 1.4s ease-in-out infinite;
  }
  .record-btn.recording .mic-icon { display: none; }
  .record-btn.recording .stop-icon { display: block; }
  @keyframes record-pulse {
    0%, 100% { box-shadow: 0 3px 0 #641a0f, 0 0 0 0 rgba(181,48,29,0.45), 0 10px 20px -8px var(--shadow-warm); }
    50%      { box-shadow: 0 3px 0 #641a0f, 0 0 0 12px rgba(181,48,29,0), 0 10px 20px -8px var(--shadow-warm); }
  }
  @media (prefers-reduced-motion: reduce) {
    .record-btn.recording { animation: none; }
  }
  .recorder .instruction {
    margin: 1rem 0 0.2rem;
    font-size: 1.05rem;
    color: var(--wood);
  }
  .recorder .hint {
    color: var(--wood-soft);
    font-size: 0.85rem;
    font-family: var(--font-mono);
    margin: 0;
  }
  .record-timer {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--record-red);
    margin: 0.9rem 0 0;
    font-variant-numeric: tabular-nums;
    display: none;
  }
  .recorder.is-recording .record-timer { display: block; }
  .recorder.is-recording .instruction { display: none; }

  .recording-preview {
    display: none;
    margin-top: 1.1rem;
    padding-top: 1rem;
    border-top: 1px dashed var(--paper-line);
    align-items: center;
    justify-content: center;
    gap: 0.85rem;
    flex-wrap: wrap;
  }
  .recording-preview.active { display: flex; }
  .recording-preview audio { max-width: 260px; }
  .btn-text {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--brick);
    background: none;
    border: 1px solid var(--brick);
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    cursor: pointer;
    transition: background .15s ease, color .15s ease;
  }
  .btn-text:hover { background: var(--brick); color: var(--manuscript); }
  .mic-permission-note {
    display: none;
    margin-top: 1rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--brick);
  }
  .mic-permission-note.active { display: block; }

  button.primary {
    font-family: var(--font-body);
    font-size: 1.05rem;
    font-weight: 600;
    background: linear-gradient(180deg, var(--brick), var(--brick-deep));
    color: var(--manuscript);
    border: none;
    border-radius: 7px;
    padding: 0.8rem 1.7rem;
    cursor: pointer;
    box-shadow: 0 3px 0 #4a170e, 0 8px 18px -8px var(--shadow-warm);
    transition: transform .12s ease, box-shadow .12s ease;
  }
  button.primary:hover:not(:disabled) { transform: translateY(-1px); }
  button.primary:active:not(:disabled) { box-shadow: 0 1px 0 #4a170e; transform: translateY(2px); }
  button.primary:disabled {
    background: #c9bba3; color: #8a7c67;
    box-shadow: none; cursor: not-allowed;
  }
  button.primary .arrow { margin-left: 0.4rem; }

  /* ---------------- Progress stepper ---------------- */
  .stepper {
    display: none;
    margin-top: 1.5rem;
    padding-top: 1.25rem;
    border-top: 1px dashed var(--paper-line);
  }
  .stepper.active { display: block; }
  .stepper ol {
    list-style: none;
    margin: 0; padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 0;
  }
  .stepper li {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: #b0a389;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .stepper li:not(:last-child)::after {
    content: "—";
    color: #d8cbb0;
    margin: 0 0.6rem;
  }
  .stepper li .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #d8cbb0;
    flex-shrink: 0;
  }
  .stepper li.done { color: var(--wood-soft); }
  .stepper li.done .dot { background: var(--copper); }
  .stepper li.current { color: var(--brick); font-weight: 600; }
  .stepper li.current .dot {
    background: var(--gold);
    animation: dot-pulse 1s ease-in-out infinite;
  }
  @keyframes dot-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(212,160,23,0.5); }
    50% { box-shadow: 0 0 0 5px rgba(212,160,23,0); }
  }

  .status {
    margin-top: 1rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--wood-soft);
  }
  .status[data-state="error"] { color: var(--brick); font-weight: 600; }

  .error-summary {
    border: 1px solid var(--brick);
    background: rgba(139,46,29,0.06);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 2rem;
  }
  .error-summary h2 {
    color: var(--brick);
    font-family: var(--font-display);
    font-size: 1.15rem;
    margin: 0 0 0.4rem;
  }
  .error-summary p { margin: 0; color: var(--wood); }

  /* ---------------- Aggregate confidence ---------------- */
  .confidence-summary {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin-bottom: 1.5rem;
  }
  .confidence-summary .figure {
    font-family: var(--font-mono);
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--brick);
    line-height: 1;
  }
  .confidence-summary .figure small {
    font-size: 1rem; color: var(--wood-soft); font-weight: 500;
  }
  .confidence-summary .track {
    flex: 1;
    height: 8px;
    border-radius: 5px;
    background: var(--manuscript-2);
    overflow: hidden;
  }
  .confidence-summary .fill {
    height: 100%;
    background: linear-gradient(90deg, var(--brick), var(--gold));
    width: 0%;
    transition: width .7s ease;
  }
  .confidence-summary .label {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--wood-soft);
  }

  /* ---------------- Transcript cards (side-by-side) ---------------- */
  .transcript-card {
    border: 1px solid var(--paper-line);
    border-radius: 9px;
    background: #FBF3E4;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .transcript-card .meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--wood-soft);
  }
  .transcript-card .timestamp {
    background: var(--wood);
    color: var(--gold-soft);
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    font-variant-numeric: tabular-nums;
  }
  .transcript-card .conf-badge {
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    background: rgba(181,101,29,0.15);
    color: var(--copper);
    font-weight: 600;
  }
  .transcript-card .conf-badge.low {
    background: rgba(139,46,29,0.15);
    color: var(--brick);
  }
  .transcript-card .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
  }
  @media (max-width: 640px) {
    .transcript-card .pair { grid-template-columns: 1fr; }
  }
  .transcript-card .col .col-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--copper);
    margin-bottom: 0.4rem;
  }
  .transcript-card .col.corrected .col-label { color: var(--brick); }
  .transcript-card .devanagari-text {
    font-family: "Noto Serif Devanagari", serif;
    font-size: 1.25rem;
    line-height: 1.85;
    color: var(--ink);
  }
  .transcript-card .col:not(.corrected) { border-right: 1px dashed var(--paper-line); padding-right: 1.25rem; }
  @media (max-width: 640px) {
    .transcript-card .col:not(.corrected) { border-right: none; padding-right: 0; border-bottom: 1px dashed var(--paper-line); padding-bottom: 0.85rem; margin-bottom: 0.85rem; }
  }
  .changed-word {
    background: rgba(212,160,23,0.28);
    border-bottom: 2px solid var(--gold);
    padding: 0 0.1em;
    border-radius: 2px;
  }

  details.changes-detail { margin-top: 1.25rem; }
  details.changes-detail summary {
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--brick);
  }
  details.changes-detail summary:hover { color: var(--brick-deep); }
  ul.change-list {
    margin: 0.85rem 0 0;
    padding-left: 0;
    list-style: none;
  }
  ul.change-list li {
    font-family: "Noto Serif Devanagari", serif;
    font-size: 1.05rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--paper-line);
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  ul.change-list li:last-child { border-bottom: none; }
  ul.change-list .from { color: var(--wood-soft); text-decoration: line-through; }
  ul.change-list .arrow { font-family: var(--font-mono); color: var(--gold); font-size: 0.85rem; }
  ul.change-list .to { color: var(--brick); font-weight: 600; }

  .empty-note {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--wood-soft);
    font-style: italic;
  }

  /* ---------------- Methodology ---------------- */
  .method-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.1rem;
  }
  .method-card {
    border-left: 3px solid var(--copper);
    padding: 0.2rem 0 0.2rem 1rem;
  }
  .method-card .step-num {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--gold);
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .method-card h3 {
    font-family: var(--font-display);
    font-size: 1.05rem;
    margin: 0.25rem 0 0.35rem;
    color: var(--wood);
  }
  .method-card p {
    margin: 0;
    font-size: 0.92rem;
    color: var(--wood-soft);
  }

  /* ---------------- Credits ---------------- */
  .credit-groups {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.5rem;
  }
  .credit-groups h3 {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--copper);
    margin: 0 0 0.7rem;
  }
  ul.credit-list { list-style: none; margin: 0; padding: 0; }
  ul.credit-list li {
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--paper-line);
    font-size: 0.95rem;
    color: var(--wood);
  }
  ul.credit-list li:last-child { border-bottom: none; }
  ul.credit-list li a { text-decoration: none; }
  ul.credit-list li a:hover { text-decoration: underline; }

  .repo-panel {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
  }
  .repo-panel code {
    font-family: var(--font-mono);
    background: var(--wood);
    color: var(--gold-soft);
    padding: 0.55rem 0.9rem;
    border-radius: 6px;
    font-size: 0.9rem;
  }
  a.repo-link {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--brick);
    text-decoration: none;
    border: 1px solid var(--brick);
    padding: 0.55rem 1rem;
    border-radius: 6px;
    transition: background .15s ease, color .15s ease;
  }
  a.repo-link:hover { background: var(--brick); color: var(--manuscript); }

  .visually-hidden {
    position: absolute; width: 1px; height: 1px;
    margin: -1px; padding: 0; border: 0;
    clip: rect(0 0 0 0); overflow: hidden;
  }

  /* ---------------- Footer ---------------- */
  footer.site-footer {
    background: linear-gradient(180deg, #2c1a10 0%, var(--wood) 100%);
    color: var(--manuscript-2);
    border-top: 3px solid var(--gold);
    margin-top: 4rem;
  }
  footer.site-footer .inner {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 2.5rem 1.25rem 3rem;
  }
  footer.site-footer .foot-word {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.3rem;
    color: var(--gold-soft);
    margin: 0 0 0.2rem;
  }
  footer.site-footer .foot-tagline {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: #cbb994;
    margin: 0 0 1.5rem;
  }
  footer.site-footer p {
    font-size: 0.88rem;
    color: #cbb994;
    max-width: 46em;
    line-height: 1.7;
    margin: 0 0 1rem;
  }
  footer.site-footer .privacy-note {
    border-left: 2px solid var(--gold);
    padding-left: 1rem;
    margin: 1.5rem 0;
  }
  footer.site-footer a { color: var(--gold-soft); }

  @media (max-width: 480px) {
    main { padding: 0 1rem 3rem; }
    .panel { padding: 1.25rem; }
    .hero { padding: 2.5rem 1rem 2rem; }
  }
</style>
</head>
<body>

<a class="skip-link" href="#main-content">Skip to main content</a>

<header class="top-bar">
  <div class="inner">
    <svg class="logo-mark" width="40" height="34" viewBox="0 0 40 34" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <!-- stepped pagoda roof -->
      <path class="roof" d="M20 1 L28 9 H12 Z" />
      <rect class="roof" x="9" y="9" width="22" height="2.4" rx="1" />
      <path class="roof" d="M20 6 L24 9 H16 Z" opacity="0.7" />
      <!-- waveform / jhya lattice bars -->
      <rect class="bar" x="6"  y="16" width="3" height="14" rx="1.4" />
      <rect class="bar" x="12" y="13" width="3" height="17" rx="1.4" />
      <rect class="bar" x="18" y="9"  width="3" height="21" rx="1.4" />
      <rect class="bar" x="24" y="13" width="3" height="17" rx="1.4" />
      <rect class="bar" x="30" y="16" width="3" height="14" rx="1.4" />
    </svg>
    <span class="wordmark">Shruti</span>
    <span class="tagline-sm">NEPAL BHASA SPEECH PLATFORM</span>
  </div>
</header>

<div class="hero">
  <div class="devanagari">श्रुति</div>
  <h1>Speech recognition and language correction for Nepal Bhasa</h1>
  <p class="lede">
    Upload a Nepal Bhasa recording, or record one directly in your browser,
    and receive a research-grade transcription, automatically corrected
    against a Newari lexicon and named-entity gazetteer, with word-level
    confidence scoring. Built for the documentation and preservation of the
    language, not as a general-purpose demo.
  </p>
  <div class="rule" aria-hidden="true"></div>
</div>

<main id="main-content">

  <section aria-labelledby="upload-heading">
    <h2 class="section-heading" id="upload-heading"><span class="num">01</span> Provide a recording</h2>
    <div class="panel">

      <div class="source-tabs" role="tablist" aria-label="Audio source">
        <button type="button" id="tab-upload" class="active" role="tab" aria-selected="true" aria-controls="panel-upload">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Upload file
        </button>
        <button type="button" id="tab-record" role="tab" aria-selected="false" aria-controls="panel-record">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          Record audio
        </button>
      </div>

      <div class="source-panel active" id="panel-upload" role="tabpanel" aria-labelledby="tab-upload">
        <label class="field-label" for="audio-input">Audio file</label>
        <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-describedby="dropzone-hint">
          <span class="glyph" aria-hidden="true">ॐ</span>
          <p class="instruction">Drag and drop a recording here, or click to choose a file</p>
          <p class="hint" id="dropzone-hint">WAV · FLAC · MP3 · OGG · M4A — up to 200MB</p>
          <p class="filename" id="filename" aria-live="polite"></p>
        </div>
        <input type="file" id="audio-input" accept=".wav,.flac,.mp3,.ogg,.m4a">
      </div>

      <div class="source-panel" id="panel-record" role="tabpanel" aria-labelledby="tab-record">
        <label class="field-label" for="record-btn">Microphone</label>
        <div class="recorder" id="recorder">
          <button type="button" class="record-btn" id="record-btn" aria-pressed="false" aria-label="Start recording">
            <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
            <svg class="stop-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>
          </button>
          <p class="instruction">Tap to start recording</p>
          <p class="record-timer" id="record-timer" aria-live="polite">0:00</p>
          <p class="hint">Speak clearly, in a quiet room if possible</p>

          <div class="recording-preview" id="recording-preview">
            <audio id="recording-audio" controls></audio>
            <button type="button" class="btn-text" id="record-again-btn">Record again</button>
          </div>

          <p class="mic-permission-note" id="mic-permission-note" role="alert">
            Microphone access was denied or is unavailable. Check your browser's
            site permissions, or use the "Upload file" tab instead.
          </p>
        </div>
      </div>

      <p style="margin-top:1.5rem;">
        <button class="primary" id="transcribe-btn" type="button" disabled>
          Transcribe recording <span class="arrow" aria-hidden="true">&rarr;</span>
        </button>
      </p>

      <p class="status" id="status" role="status" aria-live="polite"></p>

      <div class="stepper" id="stepper">
        <ol>
          <li id="step-upload"><span class="dot"></span>Uploaded</li>
          <li id="step-preprocess"><span class="dot"></span>Preprocessing audio</li>
          <li id="step-asr"><span class="dot"></span>Running speech recognition</li>
          <li id="step-correct"><span class="dot"></span>Applying Nepal Bhasa correction</li>
          <li id="step-done"><span class="dot"></span>Complete</li>
        </ol>
      </div>
    </div>
  </section>

  <div id="error-container"></div>

  <section aria-labelledby="results-heading" id="results-section" style="display:none;">
    <h2 class="section-heading" id="results-heading"><span class="num">02</span> Transcript</h2>

    <div class="panel" id="confidence-panel" style="display:none; margin-bottom:1.25rem;">
      <div class="confidence-summary">
        <div class="figure" id="agg-confidence">—<small>avg. confidence</small></div>
        <div>
          <div class="track"><div class="fill" id="agg-confidence-fill"></div></div>
          <div class="label" id="agg-confidence-detail"></div>
        </div>
      </div>
    </div>

    <div id="results-container"></div>
  </section>

  <section aria-labelledby="method-heading">
    <h2 class="section-heading" id="method-heading"><span class="num">03</span> Research methodology</h2>
    <div class="panel">
      <div class="method-grid">
        <div class="method-card">
          <div class="step-num">STAGE 1</div>
          <h3>Signal preparation</h3>
          <p>Uploads and microphone recordings alike are downmixed to mono, resampled to 16kHz, high-pass filtered, and spectrally denoised before recognition — the same signal chain used to prepare Nwāchā Munā training audio.</p>
        </div>
        <div class="method-card">
          <div class="step-num">STAGE 2</div>
          <h3>Speech segmentation</h3>
          <p>Silero VAD locates true speech/silence boundaries so long recordings are split at natural pauses, never mid-word, before being batched into the recognizer.</p>
        </div>
        <div class="method-card">
          <div class="step-num">STAGE 3</div>
          <h3>Recognition</h3>
          <p>Transcription is performed by NepConformer, a Conformer-based acoustic model fine-tuned for Nepal Bhasa, producing per-word confidence alongside text.</p>
        </div>
        <div class="method-card">
          <div class="step-num">STAGE 4</div>
          <h3>Correction</h3>
          <p>A SymSpell-based noisy-channel corrector, weighted with a KenLM language model and a proper-noun gazetteer, resolves ASR output against known Nepal Bhasa vocabulary and named entities.</p>
        </div>
      </div>
    </div>
  </section>

  <section aria-labelledby="credits-heading">
    <h2 class="section-heading" id="credits-heading"><span class="num">04</span> Credits &amp; data sources</h2>
    <div class="panel">
      <div class="credit-groups">
        <div>
          <h3>Models &amp; tooling</h3>
          <ul class="credit-list">
            <li>NVIDIA NeMo</li>
            <li>NepConformer</li>
            <li>KenLM</li>
            <li>SymSpell</li>
            <li>Silero VAD</li>
            <li>PyTorch</li>
            <li>Hugging Face</li>
          </ul>
        </div>
        <div>
          <h3>Language data</h3>
          <ul class="credit-list">
            <li>Nwāchā Munā Corpus</li>
            <li>OpenSLR&nbsp;54</li>
            <li>Newari Wikipedia — <a href="https://new.wikipedia.org" target="_blank" rel="noopener">new.wikipedia.org</a></li>
            <li>Wikimedia dataset dumps — <a href="https://dumps.wikimedia.org" target="_blank" rel="noopener">dumps.wikimedia.org</a></li>
            <li>OSCAR Corpus</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <section aria-labelledby="repo-heading">
    <h2 class="section-heading" id="repo-heading"><span class="num">05</span> Repository</h2>
    <div class="panel repo-panel">
      <code>github.com/PrageshShrestha/newari_translation</code>
      <a class="repo-link" href="https://github.com/PrageshShrestha/newari_translation" target="_blank" rel="noopener">View source &rarr;</a>
    </div>
  </section>

</main>

<footer class="site-footer">
  <div class="inner">
    <p class="foot-word">Shruti</p>
    <p class="foot-tagline">Nepal Bhasa Speech Recognition and Language Preservation Platform</p>

    <p>
      Built using NVIDIA NeMo, NepConformer, KenLM, SymSpell, Silero VAD,
      PyTorch, and Hugging Face infrastructure.
    </p>
    <p>
      Research resources include the Nwāchā Munā Corpus, OpenSLR&nbsp;54,
      Newari Wikipedia (new.wikipedia.org), the Wikimedia dataset dumps, and
      the OSCAR Corpus.
    </p>

    <div class="privacy-note">
      <p style="margin:0;">
        Audio (uploaded or recorded in-browser) is securely processed using
        cloud-hosted speech models. Temporary files are automatically
        removed after transcription completes.
      </p>
    </div>

    <p>
      Source code: <a href="https://github.com/PrageshShrestha/newari_translation" target="_blank" rel="noopener">github.com/PrageshShrestha/newari_translation</a>
    </p>
  </div>
</footer>

<script>
(function () {
  var dropzone = document.getElementById('dropzone');
  var fileInput = document.getElementById('audio-input');
  var filenameEl = document.getElementById('filename');
  var transcribeBtn = document.getElementById('transcribe-btn');
  var statusEl = document.getElementById('status');
  var stepperEl = document.getElementById('stepper');
  var errorContainer = document.getElementById('error-container');
  var resultsSection = document.getElementById('results-section');
  var resultsContainer = document.getElementById('results-container');
  var confidencePanel = document.getElementById('confidence-panel');
  var aggConfidenceEl = document.getElementById('agg-confidence');
  var aggConfidenceFillEl = document.getElementById('agg-confidence-fill');
  var aggConfidenceDetailEl = document.getElementById('agg-confidence-detail');

  // Source tabs (Upload file / Record audio)
  var tabUpload = document.getElementById('tab-upload');
  var tabRecord = document.getElementById('tab-record');
  var panelUpload = document.getElementById('panel-upload');
  var panelRecord = document.getElementById('panel-record');

  // Recorder elements
  var recorderEl = document.getElementById('recorder');
  var recordBtn = document.getElementById('record-btn');
  var recordTimerEl = document.getElementById('record-timer');
  var recordingPreview = document.getElementById('recording-preview');
  var recordingAudioEl = document.getElementById('recording-audio');
  var recordAgainBtn = document.getElementById('record-again-btn');
  var micPermissionNote = document.getElementById('mic-permission-note');

  // selectedFile always holds whatever should be POSTed to /api/transcribe --
  // it's set either from the file picker/drop, or from the mic recording
  // blob below. The rest of the transcribe flow (transcribeBtn handler)
  // doesn't need to know or care which source it came from.
  var selectedFile = null;

  var STEP_IDS = ['step-upload', 'step-preprocess', 'step-asr', 'step-correct', 'step-done'];
  var stepTimer = null;

  // ---------------- Source tab switching ----------------
  function activateTab(which) {
    var uploading = which === 'upload';
    tabUpload.classList.toggle('active', uploading);
    tabRecord.classList.toggle('active', !uploading);
    tabUpload.setAttribute('aria-selected', String(uploading));
    tabRecord.setAttribute('aria-selected', String(!uploading));
    panelUpload.classList.toggle('active', uploading);
    panelRecord.classList.toggle('active', !uploading);
    // Switching source clears whichever selection belonged to the other tab,
    // so the user can't accidentally submit a stale file from the tab they
    // just left.
    if (uploading) {
      if (recordedBlobFile) { discardRecording(); }
    } else {
      if (fileInput.value) {
        fileInput.value = '';
        filenameEl.textContent = '';
      }
    }
    updateSelection();
  }
  tabUpload.addEventListener('click', function () { activateTab('upload'); });
  tabRecord.addEventListener('click', function () { activateTab('record'); });

  function updateSelection() {
    var activeIsUpload = tabUpload.classList.contains('active');
    var file = activeIsUpload ? fileInputSelection() : recordedBlobFile;
    selectedFile = file;
    transcribeBtn.disabled = !file;
  }

  function fileInputSelection() {
    return (fileInput.files && fileInput.files.length) ? fileInput.files[0] : null;
  }

  // ---------------- File upload (drag & drop / picker) ----------------
  function setFile(file) {
    fileInput.__dtFile = file; // not used elsewhere, kept for clarity only
    filenameEl.textContent = file ? file.name : '';
    // Reflect the drop into the actual <input type=file> via DataTransfer
    // where supported, so both drag-drop and click-to-pick end up going
    // through the same fileInputSelection() path.
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    } catch (e) {
      // Safari-era fallback: DataTransfer construction can be unsupported;
      // updateSelection() below still works because we track selectedFile
      // directly in that case via the dropped file object.
      fileInput.__droppedFile = file;
    }
    updateSelection();
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
      filenameEl.textContent = fileInput.files[0].name;
    }
    updateSelection();
  });

  // ---------------- Microphone recording ----------------
  var mediaRecorder = null;
  var mediaStream = null;
  var recordedChunks = [];
  var recordedBlobFile = null;   // File object built from the recording, once stopped
  var isRecording = false;
  var recordStartTime = null;
  var recordTimerInterval = null;

  // Pick the first mimeType the browser's MediaRecorder actually supports,
  // in order of preference. Different browsers expose different encoders --
  // Chrome/Firefox/Edge support Opus-in-WebM, Safari generally only supports
  // mp4/AAC -- so this has to be checked at runtime rather than hardcoded.
  function pickSupportedMimeType() {
    var candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (var i = 0; i < candidates.length; i++) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(candidates[i])) {
        return candidates[i];
      }
    }
    return ''; // let the browser pick its own default as a last resort
  }

  function extensionForMimeType(mimeType) {
    if (mimeType.indexOf('mp4') !== -1) { return 'mp4'; }
    if (mimeType.indexOf('ogg') !== -1) { return 'ogg'; }
    return 'webm';
  }

  function formatTimer(ms) {
    var totalSec = Math.floor(ms / 1000);
    var m = Math.floor(totalSec / 60);
    var s = totalSec % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function startRecordTimer() {
    recordStartTime = Date.now();
    recordTimerEl.textContent = '0:00';
    recordTimerInterval = setInterval(function () {
      recordTimerEl.textContent = formatTimer(Date.now() - recordStartTime);
    }, 250);
  }

  function stopRecordTimer() {
    if (recordTimerInterval) { clearInterval(recordTimerInterval); recordTimerInterval = null; }
  }

  function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      micPermissionNote.classList.add('active');
      micPermissionNote.textContent = 'This browser does not support microphone recording. Use the "Upload file" tab instead.';
      return;
    }
    micPermissionNote.classList.remove('active');
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(function (stream) {
        mediaStream = stream;
        recordedChunks = [];
        var mimeType = pickSupportedMimeType();
        try {
          mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType: mimeType }) : new MediaRecorder(stream);
        } catch (e) {
          mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.addEventListener('dataavailable', function (e) {
          if (e.data && e.data.size > 0) { recordedChunks.push(e.data); }
        });

        mediaRecorder.addEventListener('stop', function () {
          var actualMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm';
          var blob = new Blob(recordedChunks, { type: actualMimeType });
          var ext = extensionForMimeType(actualMimeType);
          recordedBlobFile = new File([blob], 'recording-' + Date.now() + '.' + ext, { type: actualMimeType });

          recordingAudioEl.src = URL.createObjectURL(blob);
          recordingPreview.classList.add('active');

          // Release the microphone as soon as we're done with it -- keeping
          // the stream open after stopping would leave the browser's
          // "microphone in use" indicator on for no reason.
          mediaStream.getTracks().forEach(function (track) { track.stop(); });
          mediaStream = null;

          updateSelection();
        });

        mediaRecorder.start();
        isRecording = true;
        recorderEl.classList.add('is-recording');
        recordBtn.classList.add('recording');
        recordBtn.setAttribute('aria-pressed', 'true');
        recordBtn.setAttribute('aria-label', 'Stop recording');
        recordingPreview.classList.remove('active');
        startRecordTimer();
      })
      .catch(function () {
        micPermissionNote.classList.add('active');
        micPermissionNote.textContent = 'Microphone access was denied or is unavailable. Check your browser\'s site permissions, or use the "Upload file" tab instead.';
      });
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
    }
    isRecording = false;
    recorderEl.classList.remove('is-recording');
    recordBtn.classList.remove('recording');
    recordBtn.setAttribute('aria-pressed', 'false');
    recordBtn.setAttribute('aria-label', 'Start recording');
    stopRecordTimer();
  }

  function discardRecording() {
    recordedBlobFile = null;
    recordedChunks = [];
    recordingAudioEl.removeAttribute('src');
    recordingPreview.classList.remove('active');
    recordTimerEl.textContent = '0:00';
    updateSelection();
  }

  recordBtn.addEventListener('click', function () {
    if (isRecording) { stopRecording(); }
    else { startRecording(); }
  });

  recordAgainBtn.addEventListener('click', function () {
    discardRecording();
  });

  // ---------------- Shared: escape/format helpers ----------------
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

  function resetStepper() {
    STEP_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      el.classList.remove('done', 'current');
    });
  }

  function setStep(index, markPreviousDone) {
    STEP_IDS.forEach(function (id, i) {
      var el = document.getElementById(id);
      el.classList.remove('done', 'current');
      if (i < index) { el.classList.add('done'); }
      else if (i === index) { el.classList.add('current'); }
    });
  }

  function startStepperSimulation() {
    resetStepper();
    stepperEl.classList.add('active');
    var i = 0;
    setStep(i);
    stepTimer = setInterval(function () {
      // Advance through upload -> preprocess -> asr, then hold at "asr"
      // until the request actually resolves (we don't have a real
      // progress stream from the server, so the first three states are a
      // reasonable, honest approximation of a short pipeline that's
      // genuinely running in that order).
      if (i < 2) {
        i += 1;
        setStep(i);
      }
    }, 900);
  }

  function finishStepperSimulation() {
    if (stepTimer) { clearInterval(stepTimer); stepTimer = null; }
    setStep(3);
    setTimeout(function () { setStep(4); }, 350);
  }

  function abortStepperSimulation() {
    if (stepTimer) { clearInterval(stepTimer); stepTimer = null; }
    stepperEl.classList.remove('active');
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

  function confidenceBadge(avgConfidence) {
    if (avgConfidence === null || avgConfidence === undefined) {
      return '<span class="conf-badge">— confidence</span>';
    }
    var pct = Math.round(avgConfidence * 100);
    var cls = avgConfidence < 0.6 ? 'conf-badge low' : 'conf-badge';
    return '<span class="' + cls + '">' + pct + '% confidence</span>';
  }

  function renderResults(data) {
    resultsContainer.innerHTML = '';
    errorContainer.innerHTML = '';

    if (!data.sentences || !data.sentences.length) {
      resultsContainer.innerHTML = '<p class="empty-note">No speech was detected in this recording.</p>';
      resultsSection.style.display = 'block';
      confidencePanel.style.display = 'none';
      return;
    }

    // Aggregate confidence summary
    if (data.avg_confidence !== null && data.avg_confidence !== undefined) {
      var pct = Math.round(data.avg_confidence * 100);
      aggConfidenceEl.innerHTML = pct + '%<small>avg. confidence</small>';
      aggConfidenceFillEl.style.width = pct + '%';
      aggConfidenceDetailEl.textContent = data.sentences.length + ' sentence' +
        (data.sentences.length === 1 ? '' : 's') + ' transcribed';
      confidencePanel.style.display = 'block';
    } else {
      confidencePanel.style.display = 'none';
    }

    var html = '';
    var allChanges = [];

    data.sentences.forEach(function (s) {
      html += '<article class="transcript-card">';
      html += '<div class="meta-row">';
      html += '<span class="timestamp">' + formatTimestamp(s.start) + '</span>';
      html += confidenceBadge(s.avg_confidence);
      html += '</div>';
      html += '<div class="pair">';
      html += '<div class="col">';
      html += '<div class="col-label">As heard</div>';
      html += '<div class="devanagari-text">' + escapeHtml(s.text) + '</div>';
      html += '</div>';
      html += '<div class="col corrected">';
      html += '<div class="col-label">Corrected</div>';
      html += '<div class="devanagari-text">' + renderCorrectedWithHighlights(s.corrected_text, s.changes) + '</div>';
      html += '</div>';
      html += '</div>';
      html += '</article>';

      if (s.changes && s.changes.length) {
        allChanges = allChanges.concat(s.changes);
      }
    });

    if (allChanges.length) {
      html += '<details class="changes-detail">';
      html += '<summary>' + allChanges.length + ' correction' +
        (allChanges.length === 1 ? '' : 's') + ' applied</summary>';
      html += '<ul class="change-list">';
      allChanges.forEach(function (c) {
        html += '<li><span class="from">' + escapeHtml(c.original) + '</span>' +
          '<span class="arrow">&rarr;</span>' +
          '<span class="to">' + escapeHtml(c.corrected) + '</span></li>';
      });
      html += '</ul></details>';
    }

    resultsContainer.innerHTML = html;
    resultsSection.style.display = 'block';
  }

  function renderError(message) {
    errorContainer.innerHTML =
      '<div class="error-summary" role="alert">' +
      '<h2>There is a problem</h2>' +
      '<p>' + escapeHtml(message) + '</p>' +
      '</div>';
    resultsSection.style.display = 'none';
    resultsContainer.innerHTML = '';
  }

  // ---------------- Submit: identical for both a picked/dropped file and
  // a mic recording -- selectedFile is whichever the active tab produced. ----
  transcribeBtn.addEventListener('click', function () {
    if (!selectedFile) { return; }
    // If a recording is still in progress when the button is somehow
    // reachable, stop it first so the blob is finalized before upload.
    if (isRecording) { stopRecording(); }

    transcribeBtn.disabled = true;
    statusEl.removeAttribute('data-state');
    statusEl.textContent = 'Transcribing recording. This may take a moment.';
    errorContainer.innerHTML = '';
    resultsSection.style.display = 'none';
    resultsContainer.innerHTML = '';
    startStepperSimulation();

    var formData = new FormData();
    formData.append('audio', selectedFile, selectedFile.name);

    fetch('/api/transcribe', { method: 'POST', body: formData })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, data: data };
        });
      })
      .then(function (result) {
        transcribeBtn.disabled = false;
        if (!result.ok || result.data.error) {
          abortStepperSimulation();
          statusEl.setAttribute('data-state', 'error');
          statusEl.textContent = 'Transcription failed.';
          renderError(result.data.error || 'An unknown error occurred.');
          return;
        }
        finishStepperSimulation();
        statusEl.textContent = 'Transcription complete.';
        renderResults(result.data);
      })
      .catch(function (err) {
        transcribeBtn.disabled = false;
        abortStepperSimulation();
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