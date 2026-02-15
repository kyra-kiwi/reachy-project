"""Test script for Piper TTS with British voice.

Piper TTS is a fast, local neural text-to-speech system.
For Mac M2, you can install it via:

uv pip install piper-tts   
python -m piper.download_voices en_GB-southern_english_female-low

British voice  are available from:
  https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/southern_english_female/low

"""


from pathlib import Path
from piper import PiperVoice
import wave

def test_piper_british_voice():
    """Test Piper TTS with British voice."""
    
    # Test text
    test_text = "Hello, this is a test of Piper text to speech with a British accent. How are you today?"
    # Output file
    output_file = "piper_british_test.wav"

    voice = PiperVoice.load("./en_GB-southern_english_female-low.onnx")
    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(test_text, wav_file)
    return True

if __name__ == "__main__":
    success = test_piper_british_voice()
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test did not complete. See instructions above.")
