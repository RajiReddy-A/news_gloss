import os
import sys
from pathlib import Path

# Configure Matplotlib config dir to avoid sandbox font read errors
os.environ["MPLCONFIGDIR"] = "/tmp/cache"

from dotenv import load_dotenv
load_dotenv()

from pipeline.tts import synthesise

# Test Telugu
print("Step 1: Testing Telugu Speech Synthesis...")
te_text = "నమస్కారం, ఇది ఒక పరీక్ష."
te_output = Path("test_synthesis_te.wav")

try:
    path, engine = synthesise(te_text, "te", te_output)
    print(f"Telugu SUCCESS: Generated using {engine} at {path}")
    if te_output.exists():
        print(f"File size: {te_output.stat().st_size} bytes")
        te_output.unlink() # clean up
except Exception as e:
    print(f"Telugu FAILED: {type(e).__name__}: {e}")

# Test Hindi
print("\nStep 2: Testing Hindi Speech Synthesis...")
hi_text = "नमस्ते, यह एक परीक्षण है।"
hi_output = Path("test_synthesis_hi.wav")

try:
    path, engine = synthesise(hi_text, "hi", hi_output)
    print(f"Hindi SUCCESS: Generated using {engine} at {path}")
    if hi_output.exists():
        print(f"File size: {hi_output.stat().st_size} bytes")
        hi_output.unlink() # clean up
except Exception as e:
    print(f"Hindi FAILED: {type(e).__name__}: {e}")
