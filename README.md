markdown

# Shruti - Newari ASR + Autocorrect Application

Unified local web application that provides speech-to-text transcription and autocorrection for Newari (Nepal Bhasa) audio files.

## Architecture Overview

The application consists of three main components:

1. **ASR Engine**: NeMo-based Conformer model (NwachaMuna-NepConformer-Aug) for speech recognition
2. **Correction Engine**: SymSpell + noisy-channel autocorrector with vocabulary and bigram language model
3. **Web Interface**: Flask-based single-page application with drag-and-drop audio upload

## Artifacts

The correction engine requires three binary files in the `artifacts/` directory:

newari_asr_app/
artifacts/
dictionary.bin # Word list with quantized unigram log-probabilities
symspell_index.bin # Delete-variant lookup index for SymSpell
bigrams.bin # Word pair frequency counts (word_id_a, word_id_b, count)
text


Optional file for gazetteer support:

gazetteer_everestner.txt # Proper-noun entities from EverestNER
text


Optional file for confusion-based correction:

word_substitution_pairs.csv # Observed ASR substitution patterns from evaluation
text


## Installation

Install required dependencies:

```bash
pip install flask nemo_toolkit[asr] omegaconf torch soundfile scipy

Running the Application

Start the web server:
bash

cd newari_asr_app
python run_browser.py

Navigate to http://127.0.0.1:5000 in a web browser.
Usage

    Drag an audio file (.wav, .flac, .mp3, .ogg, .m4a) onto the drop zone or click to select a file

    Click "Transcribe & Correct"

    The application displays:

        Raw ASR output (left panel)

        Autocorrected text (right panel)

        List of corrections applied

        Average confidence score

Correction Engine

The CorrectionEngine class provides text correction functionality:
python

from correction_engine import CorrectionEngine

# Initialize with artifacts directory
engine = CorrectionEngine(artifacts_dir="artifacts")

# Optional: load confusion pairs from evaluation run
engine = CorrectionEngine(
    artifacts_dir="artifacts",
    confusion_csv="word_substitution_pairs.csv"
)

# Correct text
corrected_text, changes = engine.correct_text("misrecognized text")

Correction Pipeline

    Vocabulary Check: Words already in the training vocabulary are preserved unchanged (prevents over-correction)

    SymSpell Lookup: Unknown words are looked up via delete-variant index

    Weighted Edit Distance: Candidates are scored using phonetically-aware substitution costs

    Language Model Scoring: Unigram and bigram probabilities from training corpus

    Confusion-based Correction (optional): Applies observed ASR substitution patterns as a second pass

Configuration Parameters

    MAX_EDIT_DISTANCE = 2: Maximum edit distance for SymSpell lookup

    LAMBDA = 3.0: Weight of error cost versus language model score

    CONFUSABLE_GROUPS: Phonetically similar character groups for weighted edit distance

    min_count = 10: Minimum frequency threshold for confusion pairs

Audio Processing
Preprocessing

    Audio is resampled to 16kHz mono (ASR model requirement)

    Multi-channel audio is downmixed to mono

    Supported formats: WAV, FLAC, MP3, OGG, M4A

Segmentation

Long audio files are split using Silero VAD (Voice Activity Detection):

    Speech regions are detected and merged if separated by short pauses

    Minimum speech segment length: 0.2 seconds

    Maximum chunk length: 15 seconds (safety cap)

    VAD fallback: fixed-window splitting if VAD unavailable

File Structure
text

newari_asr_app/
├── run_browser.py           # Flask web application
├── correction_engine.py     # Standalone correction engine
├── artifacts/               # Required binary artifacts
│   ├── dictionary.bin
│   ├── symspell_index.bin
│   └── bigrams.bin
├── uploads/                 # Temporary audio storage (auto-cleaned)
└── README.md

Evaluation

For performance evaluation and confusion pair generation, use the companion script:
bash

python evaluate_asr_and_correction.py \
    --data-dir ./datasets \
    --artifacts-dir ./newari_asr_app/artifacts \
    --asr-model "ilprl-docse/NwachaMuna-NepConformer-Aug" \
    --output-dir ./results \
    --splits train validation test

The evaluation script produces:

    word_substitution_pairs.csv: Observed ASR confusion patterns

    char_substitution_pairs.csv: Character-level confusion patterns

    per_utterance_results.csv: Detailed utterance-level metrics

    corpus_summary.json: Aggregate WER/CER statistics

Dependencies

    Flask (web server)

    NeMo Toolkit (ASR model)

    SoundFile (audio I/O)

    SciPy (resampling)

    PyTorch (deep learning framework)

    Silero VAD (voice activity detection)

License

This project is for research and development purposes. ASR model weights and training data are subject to their respective licenses.
Development Notes

The correction_engine.py module can be used independently of the web application for batch processing or integration with other services. The engine loads all artifacts once at initialization and maintains them in memory for fast inference.
