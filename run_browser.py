"""
run_browser.py

Unified local web app:

    audio file (drag & drop) --> Whisper Nepali ASR (Dragneel/whisper-large-v3-nepali-openslr)
                              --> Newari autocorrect (correction_engine.py)
                              --> rendered side-by-side in the browser

Long clips are split into CHUNK_SECONDS-long segments before being fed to
the ASR model (processed sequentially), and the per-chunk outputs are 
stitched back together before autocorrect runs on the full text.

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
from scipy.signal import resample_poly
from math import gcd
from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

from correction_engine import CorrectionEngine

# Whisper models expect 16kHz mono input
ASR_TARGET_SAMPLE_RATE = 16000

# Long recordings are split into consecutive chunks of this length
# Whisper models have a context window of ~30 seconds, so we use 15 for safety
CHUNK_SECONDS = 15

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
_asr_pipeline = None
_corrector = None

# Model selection - use the best open-source Nepali ASR model
# Options:
# - "Dragneel/whisper-large-v3-nepali-openslr" - Best accuracy (1.55B params)
# - "sumanpaudel1997/nepali-asr-whisper-medium" - Good accuracy (769M params)  
# - "Dragneel/whisper-small-nepali" - Fastest (244M params)
MODEL_ID = "Dragneel/whisper-large-v3-nepali-openslr"

def get_corrector():
    global _corrector
    if _corrector is None:
        _corrector = CorrectionEngine(ARTIFACTS_DIR)
    return _corrector

def get_asr_pipeline():
    """Lazy-load the Whisper model pipeline."""
    global _asr_pipeline
    if _asr_pipeline is None:
        print(f"Loading ASR model: {MODEL_ID}...")
        
        # Check if CUDA is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {device}")
        
        # Create the pipeline with optimized settings
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
            model_kwargs={"use_safetensors": True},
            generate_kwargs={
                "task": "transcribe",
                "language": "ne",  # Nepali
                "temperature": 0.0,
                "no_repeat_ngram_size": 3
            }
        )
        print("ASR model loaded successfully!")
    
    return _asr_pipeline


def get_asr_model():
    """Wrapper for compatibility with existing code that expects a model object."""
    # Return the pipeline directly
    return get_asr_pipeline()


# ---------------------------------------------------------------------
# Audio preprocessing (mono + 16kHz), required before every ASR call
# ---------------------------------------------------------------------

def preprocess_audio_for_asr(input_path: str, output_path: str,
                              target_sr: int = ASR_TARGET_SAMPLE_RATE) -> None:
    """Reads any audio soundfile can open, downmixes to mono if needed,
    resamples to target_sr if needed, and writes a clean 16-bit PCM wav to
    output_path."""
    data, sr = sf.read(input_path, always_2d=True)  # shape: (frames, channels)

    # Downmix to mono by averaging channels
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to the model's expected sample rate if needed
    if sr != target_sr:
        g = gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        data = resample_poly(data, up, down)

    data = data.astype(np.float32)
    sf.write(output_path, data, target_sr, subtype="PCM_16")


# ---------------------------------------------------------------------
# Chunking for long audio: split into CHUNK_SECONDS segments
# ---------------------------------------------------------------------

def chunk_audio(processed_path: str, chunk_dir: str, tag: str,
                 chunk_seconds: int = CHUNK_SECONDS,
                 sr: int = ASR_TARGET_SAMPLE_RATE):
    """Splits an already-preprocessed (mono, sr Hz) wav into consecutive
    chunk_seconds-long wav files. Returns a list of chunk file paths in order."""
    data, file_sr = sf.read(processed_path, always_2d=False)
    if file_sr != sr:
        raise ValueError(f"chunk_audio expects {sr}Hz audio, got {file_sr}Hz")

    total_samples = len(data)
    chunk_samples = chunk_seconds * sr

    if total_samples <= chunk_samples:
        return [processed_path]

    chunk_paths = []
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        chunk_data = data[start:end].astype(np.float32)
        chunk_path = os.path.join(chunk_dir, f"chunk_{tag}_{i}.wav")
        sf.write(chunk_path, chunk_data, sr, subtype="PCM_16")
        chunk_paths.append(chunk_path)
    return chunk_paths


# ---------------------------------------------------------------------
# Helper: Post-process ASR output (optional transliteration)
# ---------------------------------------------------------------------

def post_process_asr_output(text: str) -> str:
    """Clean up common ASR artifacts and fix common phonetic errors."""
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Common phonetic corrections for Newari (add more as you discover them)
    corrections = {
        # If the ASR consistently gets certain words wrong, add them here
        # Example: 'जत' : 'जक',
        #          'छगू' : 'छगु',
    }
    
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    
    return text


# ---------------------------------------------------------------------
# API Routes
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
    chunk_paths = []
    
    try:
        preprocess_audio_for_asr(path, processed_path)
    except Exception as e:
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"error": f"Could not read/convert audio file: {e}"}), 400

    try:
        pipeline = get_asr_pipeline()
        corrector = get_corrector()

        # Split into chunks if needed
        chunk_paths = chunk_audio(processed_path, UPLOAD_DIR, tag=str(ts))

        # Transcribe each chunk and collect results
        chunk_texts = []
        chunk_confidences = []
        
        print(f"Transcribing {len(chunk_paths)} chunk(s)...")
        
        for i, chunk_path in enumerate(chunk_paths):
            print(f"  Processing chunk {i+1}/{len(chunk_paths)}...")
            
            # Transcribe the chunk
            result = pipeline(
                chunk_path,
                return_timestamps=False
            )
            
            text = result.get("text", "").strip()
            chunk_texts.append(text)
            
            # Try to extract confidence (Whisper doesn't always provide this)
            if "confidence" in result:
                chunk_confidences.append(result["confidence"])
        
        # Stitch transcripts back together
        raw_text = " ".join(chunk_texts).strip()
        
        # Apply post-processing (optional transliteration)
        raw_text = post_process_asr_output(raw_text)

        # Calculate average confidence if available
        avg_confidence = None
        if chunk_confidences:
            avg_confidence = float(sum(chunk_confidences) / len(chunk_confidences))

        # Generate word-level confidence (simplified - Whisper doesn't provide per-word)
        words = raw_text.split()
        word_conf_pairs = [{"word": w, "confidence": avg_confidence} for w in words]

        # Run autocorrect
        corrected_text, changes = corrector.correct_text(raw_text)

        return jsonify({
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "changes": changes,
            "word_confidence": word_conf_pairs,
            "avg_confidence": avg_confidence,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary files
        cleanup_paths = [path, processed_path]
        if chunk_paths != [processed_path]:
            cleanup_paths += chunk_paths
        for p in cleanup_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------
# Front end (single-file template)
# ---------------------------------------------------------------------

INDEX_HTML = r"""
<!doctype html>
<html lang="ne">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>श्रुति — Newari ASR + Autocorrect</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+Devanagari:wght@400;600;700&family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg: #170f1f;
    --panel: #221731;
    --panel-2: #2b1c3d;
    --gold: #c9a24b;
    --gold-soft: #e4c877;
    --vermilion: #b23a48;
    --cream: #f2e8d5;
    --muted: #a493b0;
    --line: rgba(201,162,75,0.25);
  }
  *{ box-sizing: border-box; }
  html,body{ height:100%; }
  body{
    margin:0;
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(178,58,72,0.18), transparent 60%),
      radial-gradient(900px 500px at 100% 10%, rgba(201,162,75,0.10), transparent 55%),
      var(--bg);
    color: var(--cream);
    font-family: 'Spectral', serif;
    min-height:100%;
    padding: 48px 20px 80px;
  }
  .wrap{ max-width: 920px; margin: 0 auto; }

  header{ text-align:center; margin-bottom: 46px; }
  .eyebrow{
    font-family:'JetBrains Mono', monospace;
    letter-spacing: .22em;
    text-transform: uppercase;
    font-size: 11px;
    color: var(--gold);
    margin-bottom: 14px;
  }
  h1{
    font-family:'Noto Serif Devanagari', serif;
    font-weight:700;
    font-size: 44px;
    margin: 0 0 10px;
    color: var(--cream);
  }
  h1 span{ color: var(--gold-soft); font-family:'Spectral', serif; font-style:italic; font-weight:500; font-size: 22px; }
  .sub{ color: var(--muted); font-size: 16px; max-width: 560px; margin: 0 auto; line-height:1.6; }

  .leaf-frame{
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 6px;
    background: linear-gradient(180deg, rgba(201,162,75,0.06), transparent);
    margin-bottom: 18px;
  }
  .dropzone{
    position: relative;
    border: 1.5px dashed rgba(201,162,75,0.5);
    border-radius: 14px;
    padding: 46px 20px;
    text-align:center;
    cursor:pointer;
    transition: border-color .2s, background .2s;
    background: var(--panel);
  }
  .dropzone:hover, .dropzone.drag{
    border-color: var(--gold-soft);
    background: var(--panel-2);
  }
  .dropzone .glyph{ font-size: 34px; color: var(--gold); display:block; margin-bottom: 10px; }
  .dropzone .title{ font-size: 17px; color: var(--cream); margin-bottom: 6px; }
  .dropzone .hint{ font-size: 13px; color: var(--muted); font-family:'JetBrains Mono', monospace; }
  .dropzone input{ display:none; }

  .filebar{
    display:none;
    align-items:center;
    justify-content:space-between;
    gap: 14px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 22px;
    font-family:'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--muted);
  }
  .filebar strong{ color: var(--cream); font-weight:500; }
  .filebar button{
    background: var(--gold);
    color: #170f1f;
    border:none;
    border-radius: 8px;
    padding: 10px 18px;
    font-family:'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing:.05em;
    text-transform:uppercase;
    cursor:pointer;
    transition: background .2s, transform .15s;
  }
  .filebar button:hover{ background: var(--gold-soft); }
  .filebar button:active{ transform: scale(.97); }
  .filebar button:disabled{ opacity:.5; cursor:not-allowed; }

  .loader{
    display:none;
    align-items:center;
    justify-content:center;
    gap: 14px;
    padding: 30px 0;
    color: var(--gold-soft);
    font-family:'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing:.08em;
  }
  .chakra{
    width: 22px; height:22px;
    border: 2px solid rgba(201,162,75,0.25);
    border-top-color: var(--gold);
    border-radius:50%;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin{ to{ transform: rotate(360deg); } }

  .error{
    display:none;
    background: rgba(178,58,72,0.15);
    border: 1px solid rgba(178,58,72,0.4);
    color: #f3c9ce;
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 22px;
  }

  .results{ display:none; }

  .confidence-row{
    display:flex;
    align-items:center;
    gap: 14px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 22px;
    font-family:'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing:.05em;
    color: var(--muted);
  }
  .confidence-track{
    flex:1;
    height: 6px;
    background: rgba(201,162,75,0.15);
    border-radius: 4px;
    overflow:hidden;
  }
  .confidence-fill{
    height:100%;
    background: linear-gradient(90deg, var(--vermilion), var(--gold-soft));
    width:0%;
    transition: width .4s ease;
  }
  #confValue{ color: var(--gold-soft); min-width: 40px; text-align:right; }

  .panels{
    display:grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-bottom: 22px;
  }
  @media (max-width: 720px){ .panels{ grid-template-columns: 1fr; } }

  .panel{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
  }
  .panel .label{
    font-family:'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 14px;
  }
  .panel .body-text{
    font-family:'Noto Serif Devanagari', serif;
    font-size: 20px;
    line-height: 1.9;
    color: var(--cream);
  }
  .panel.corrected .body-text .fix{
    color: var(--gold-soft);
    border-bottom: 1px dashed var(--gold);
    padding-bottom: 1px;
  }
  .panel.raw .body-text .low-conf{
    color: #f3c9ce;
    border-bottom: 1px dotted var(--vermilion);
  }

  .changes{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
  }
  .changes .label{
    font-family:'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 14px;
  }
  .changes .empty{ color: var(--muted); font-size: 14px; font-style: italic; }
  .change-item{
    display:flex;
    align-items:center;
    gap: 10px;
    font-family:'Noto Serif Devanagari', serif;
    font-size: 18px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(201,162,75,0.12);
  }
  .change-item:last-child{ border-bottom:none; }
  .change-item .from{ color: var(--muted); text-decoration: line-through; }
  .change-item .arrow{ color: var(--gold); font-family:'JetBrains Mono', monospace; font-size:13px; }
  .change-item .to{ color: var(--gold-soft); }

  footer{
    text-align:center;
    margin-top: 50px;
    color: var(--muted);
    font-family:'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .06em;
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Whisper Nepali ASR &middot; SymSpell + Noisy Channel</div>
    <h1>श्रुति <span>· Shruti</span></h1>
    <p class="sub">Drop a Newari (Nepal Bhasa) audio clip. It's transcribed by the best open-source Nepali ASR model,
      then run through the dictionary-and-bigram autocorrector, side by side.</p>
  </header>

  <div class="leaf-frame">
    <div class="dropzone" id="dropzone">
      <span class="glyph">ॐ</span>
      <div class="title">Drag audio here, or click to choose a file</div>
      <div class="hint">.wav &middot; .flac &middot; .mp3 &middot; .ogg &middot; .m4a</div>
      <input type="file" id="fileInput" accept=".wav,.flac,.mp3,.ogg,.m4a">
    </div>
  </div>

  <div class="filebar" id="filebar">
    <span>Selected: <strong id="fileName">—</strong></span>
    <button id="transcribeBtn">Transcribe &amp; Correct</button>
  </div>

  <div class="loader" id="loader">
    <div class="chakra"></div>
    <div class="msg" id="loaderMsg">RUNNING ASR MODEL…</div>
  </div>

  <div class="error" id="errorBox"></div>

  <div class="results" id="results">

    <div class="confidence-row" id="confRow" style="display:none;">
      <span>AVG CONFIDENCE</span>
      <div class="confidence-track"><div class="confidence-fill" id="confFill"></div></div>
      <span id="confValue">—</span>
    </div>

    <div class="panels">
      <div class="panel raw">
        <div class="label">As Heard &middot; ASR Output</div>
        <div class="body-text" id="rawText">—</div>
      </div>
      <div class="panel corrected">
        <div class="label">As Written &middot; Autocorrected</div>
        <div class="body-text" id="correctedText">—</div>
      </div>
    </div>

    <div class="changes">
      <div class="label">Corrections Applied</div>
      <div id="changesList"><div class="empty">No changes yet.</div></div>
    </div>

  </div>

  <footer>श्रुति · running locally · nothing leaves this machine</footer>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const filebar = document.getElementById('filebar');
const fileNameEl = document.getElementById('fileName');
const transcribeBtn = document.getElementById('transcribeBtn');
const loader = document.getElementById('loader');
const loaderMsg = document.getElementById('loaderMsg');
const errorBox = document.getElementById('errorBox');
const results = document.getElementById('results');
const rawTextEl = document.getElementById('rawText');
const correctedTextEl = document.getElementById('correctedText');
const changesList = document.getElementById('changesList');
const confRow = document.getElementById('confRow');
const confFill = document.getElementById('confFill');
const confValue = document.getElementById('confValue');

let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());

['dragenter','dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('drag'); })
);
['dragleave','drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('drag'); })
);
dropzone.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) setFile(f);
});

function setFile(f){
  selectedFile = f;
  fileNameEl.textContent = f.name;
  filebar.style.display = 'flex';
  errorBox.style.display = 'none';
}

transcribeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  errorBox.style.display = 'none';
  results.style.display = 'none';
  loader.style.display = 'flex';
  transcribeBtn.disabled = true;
  loaderMsg.textContent = 'RUNNING ASR MODEL…';

  const fd = new FormData();
  fd.append('audio', selectedFile);

  try{
    const resp = await fetch('/api/transcribe', { method:'POST', body: fd });
    const data = await resp.json();
    loader.style.display = 'none';
    transcribeBtn.disabled = false;

    if (!resp.ok || data.error){
      errorBox.textContent = data.error || 'Something went wrong.';
      errorBox.style.display = 'block';
      return;
    }
    renderResults(data);
  }catch(err){
    loader.style.display = 'none';
    transcribeBtn.disabled = false;
    errorBox.textContent = 'Request failed: ' + err.message;
    errorBox.style.display = 'block';
  }
});

function escapeHtml(s){
  return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function renderResults(data){
  // Raw text, with low-confidence words underlined
  if (data.word_confidence && data.word_confidence.length){
    rawTextEl.innerHTML = data.word_confidence.map(wc => {
      const w = escapeHtml(wc.word);
      if (wc.confidence !== null && wc.confidence < 0.6){
        return `<span class="low-conf">${w}</span>`;
      }
      return w;
    }).join(' ');
  } else {
    rawTextEl.textContent = data.raw_text || '(empty)';
  }

  // Corrected text, with changed words highlighted
  const changedSet = new Set((data.changes || []).map(c => c.corrected));
  const correctedWords = (data.corrected_text || '').split(' ');
  correctedTextEl.innerHTML = correctedWords.map(w => {
    const esc = escapeHtml(w);
    return changedSet.has(w) ? `<span class="fix">${esc}</span>` : esc;
  }).join(' ');

  // Change list
  if (data.changes && data.changes.length){
    changesList.innerHTML = data.changes.map(c =>
      `<div class="change-item">
         <span class="from">${escapeHtml(c.original)}</span>
         <span class="arrow">&rarr;</span>
         <span class="to">${escapeHtml(c.corrected)}</span>
       </div>`
    ).join('');
  } else {
    changesList.innerHTML = '<div class="empty">No corrections needed — every word matched the dictionary.</div>';
  }

  // Confidence bar
  if (data.avg_confidence !== null && data.avg_confidence !== undefined){
    confRow.style.display = 'flex';
    const pct = Math.round(data.avg_confidence * 100);
    confFill.style.width = pct + '%';
    confValue.textContent = pct + '%';
  } else {
    confRow.style.display = 'none';
  }

  results.style.display = 'block';
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 60)
    print("श्रुति (Shruti) - Newari Speech Recognition System")
    print("=" * 60)
    print(f"ASR Model: {MODEL_ID}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nStarting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)