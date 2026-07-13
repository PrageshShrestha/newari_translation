"""
Gradio version of Shruti — Nepal Bhasa Speech Recognition
Run with: python app_gradio.py
"""

import os
import tempfile
import time
import traceback
import gradio as gr
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt, stft, istft
from math import gcd

# Import from existing run_browser
from run_browser import (
    get_asr_model, 
    get_corrector, 
    preprocess_audio_for_asr,
    enhance_audio_for_asr,
    ASR_TARGET_SAMPLE_RATE,
    CHUNK_SECONDS
)

# ---------------------------------------------------------------------
# Transcription function (adapted from run_browser.py)
# ---------------------------------------------------------------------
def transcribe_audio(audio_file):
    """
    Takes an audio file path, runs ASR + correction, returns transcript.
    """
    if audio_file is None:
        return "No audio file provided.", "No audio file provided."

    try:
        # Load models (lazy-loaded)
        model = get_asr_model()
        corrector = get_corrector()

        # Preprocess audio (resample to 16kHz mono, enhance)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            processed_path = tmp.name
        
        preprocess_audio_for_asr(audio_file, processed_path)

        # Run ASR
        hypotheses = model.transcribe([processed_path], return_hypotheses=True)
        
        if not hypotheses:
            return "No speech detected.", "No speech detected."

        raw_text = hypotheses[0].text or ""

        # Run correction
        corrected_text, changes = corrector.correct_text(raw_text)

        # Clean up temp file
        try:
            os.remove(processed_path)
        except OSError:
            pass

        return raw_text, corrected_text

    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", f"Error: {str(e)}"

# ---------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(
        label="Upload Nepal Bhasa Recording",
        type="filepath"
    ),
    outputs=[
        gr.Textbox(label="As Heard (Raw ASR)", lines=8),
        gr.Textbox(label="Corrected (With Post-Processing)", lines=8)
    ],
    title="Shruti — Nepal Bhasa Speech Recognition",
    description="Upload a Nepal Bhasa (Newari) recording to transcribe and correct.",
    examples=[],
    theme="soft",
    cache_examples=False,
)

# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True for a public link even without Spaces
    )

