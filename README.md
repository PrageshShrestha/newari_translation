# श्रुति (Shruti) — Newari ASR + Autocorrect, unified local app

One script (`run_browser.py`) that runs a small local website:
drag an audio clip in → NeMo ASR transcribes it → the notebook's
SymSpell + noisy-channel autocorrector cleans it up → both versions,
plus per-word confidence, are shown side by side.

## 1. Put your exported artifacts here

Copy the three files the notebook already generated for you into `artifacts/`:

```
newari_asr_app/
  artifacts/
    dictionary.bin
    symspell_index.bin
    bigrams.bin
```

## 2. Install dependencies

```bash
pip install flask nemo_toolkit[asr] omegaconf torch
```

(`nemo_toolkit[asr]` is the same package your ASR snippet already depends on —
skip this step if your environment already has it.)

## 3. Run it

```bash
cd newari_asr_app
python run_browser.py
```

Then open **http://127.0.0.1:5000** in your browser.

- Drag a `.wav` / `.flac` / `.mp3` / `.ogg` / `.m4a` file onto the drop zone
  (or click it to pick a file), then click **Transcribe & Correct**.
- The ASR model loads once, the first time you transcribe (this can take a
  little while); after that, requests are fast.
- Left panel: raw ASR output, with any word the model was under ~60%
  confident on underlined in red.
- Right panel: the autocorrected version, with every word the corrector
  changed highlighted in gold.
- Below that: an explicit list of every correction made (original → fixed),
  and an average-confidence bar for the transcription.

## How it's wired together

- `correction_engine.py` is a standalone reader for the three `.bin`
  artifacts — it re-implements the exact same `deletes()`,
  `weighted_edit_distance()`, `lm_logprob()`, and `correct_text()` logic
  from the notebook, just reading from the compact binary files instead of
  keeping the training-time Python dictionaries in memory. You can also
  `import correction_engine` and use it on its own, outside the web app.
- `run_browser.py` is a single Flask app: it loads the ASR model and the
  `CorrectionEngine` once at first use, serves the UI, and exposes one
  endpoint, `POST /api/transcribe`, that a dropped file is sent to and that
  returns JSON with the raw text, corrected text, per-word confidences, and
  the list of changes. Everything stays on your machine — no external calls
  besides loading fonts from Google Fonts.

## Notes

- If `dictionary.bin` / `symspell_index.bin` / `bigrams.bin` are missing from
  `artifacts/`, `CorrectionEngine` will raise a clear `FileNotFoundError`
  when the first request comes in — check the path in `ARTIFACTS_DIR` at the
  top of `run_browser.py` if you keep them somewhere else.
- Uploaded audio is saved to `uploads/` only for the duration of processing
  and deleted immediately after (success or failure).
- If your NeMo model version doesn't populate `hyp.word_confidence`, the app
  just skips the confidence highlighting/bar and shows the transcript as-is.
