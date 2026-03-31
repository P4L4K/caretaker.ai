import os
import base64
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

# Check for Google Cloud credentials
# The user should set GOOGLE_APPLICATION_CREDENTIALS in their .env
# as the path to their JSON key file.

def generate_speech_base64(text: str, language_code: str = "hi-IN") -> str:
    """
    Converts text to speech using Google Cloud TTS Neural2 models.
    Returns the audio content as a base64 encoded string.
    """
    try:
        # Initialize the client
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Voice selection based on language code
        # Preferred: Neural2 models for high quality
        if language_code.startswith("hi"):
            voice_name = "hi-IN-Neural2-A"  # Female, clear and natural
        else:
            voice_name = "en-US-Neural2-F"  # Female, professional and warm

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        # Audio configuration
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        # API Call
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Encode to base64
        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
        return audio_base64

    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""

def is_hindi(text: str) -> bool:
    """Detects if text contains Devanagari characters."""
    return any("\u0900" <= char <= "\u097f" for char in text)
