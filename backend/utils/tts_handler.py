import os
import base64
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

_tts_unavailable = False  # circuit breaker: True after first billing/403 error

def generate_speech_base64(text: str, language_code: str = "hi-IN") -> str:
    global _tts_unavailable
    if _tts_unavailable:
        return ""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        if language_code.startswith("hi"):
            voice_name = "hi-IN-Neural2-A"
        else:
            voice_name = "en-US-Neural2-F"

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode("utf-8")

    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "BILLING_DISABLED" in error_str or "billing" in error_str.lower():
            _tts_unavailable = True
            print("[TTS] Billing not enabled — TTS disabled for this session")
        else:
            print(f"[TTS Error] {e}")
        return ""

def is_hindi(text: str) -> bool:
    """Detects if text contains Devanagari characters."""
    return any("\u0900" <= char <= "\u097f" for char in text)
