"""
Socket.IO manager — mirrors the WebSocket event handlers from
coughandsneezedetection/app.py, ported to python-socketio (ASGI).
"""

import socketio
import logging
import datetime
from typing import Dict, Any

from services.audio_detection import audio_service
from config import SessionLocal
from tables.audio_events import AudioEvent, AudioEventType
from repository.users import JWTRepo

logger = logging.getLogger(__name__)

# Create Socket.IO server
sio_server = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8
)

# Create ASGI app
sio_app = socketio.ASGIApp(sio_server)


@sio_server.event
async def connect(sid, environ):
    audio_service.reset_buffer()
    logger.info(f"  [+] Client connected: {sid}")
    await sio_server.emit("connection_ack", {"status": "connected"}, room=sid)


@sio_server.event
async def disconnect(sid):
    audio_service.reset_buffer()
    logger.info(f"  [-] Client disconnected: {sid}")


@sio_server.event
async def reset(sid):
    """Clear server-side buffer and acknowledge — mirrors original handle_reset()."""
    audio_service.reset_buffer()
    await sio_server.emit("reset_ack", {
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }, room=sid)
    logger.info("  [RESET] State reset")


@sio_server.event
async def authenticate(sid, data):
    """
    Receive JWT token and care_recipient_id from client.
    Mirrors original handle_authenticate().
    """
    token = data.get("token")
    care_recipient_id = data.get("care_recipient_id") or data.get("recipient_id")

    if not token:
        await sio_server.emit("auth_error", {"detail": "Missing token"}, room=sid)
        return

    try:
        payload = JWTRepo.decode_token(token)
        if not payload:
            await sio_server.emit("auth_error", {"detail": "Invalid token"}, room=sid)
            return

        username = payload.get("sub")

        await sio_server.save_session(sid, {
            "username": username,
            "care_recipient_id": care_recipient_id,
            "authenticated": True
        })

        logger.info(f"  [AUTH] {sid} authenticated as {username}, recipient={care_recipient_id}")
        await sio_server.emit("auth_ack", {"status": "authenticated"}, room=sid)

    except Exception as e:
        logger.error(f"Authentication failed for {sid}: {e}")
        await sio_server.emit("auth_error", {"detail": str(e)}, room=sid)


@sio_server.event
async def audio_chunk(sid, data):
    """
    Receive raw PCM audio bytes, process through sliding window,
    and emit classification + alert events.

    Mirrors original handle_audio_chunk() exactly.
    """
    try:
        data_len = len(data) if data else 0
        # logger.info(f"  [CHUNK] Received {data_len} bytes from {sid}")
        # process_audio_chunk returns a list of results (one per window)
        results = audio_service.process_audio_chunk(data)

        for result in results:
            # Always emit classification so all frontends update (dashboard and monitor)
            await sio_server.emit("classification", result)

            predicted  = result.get("predicted")
            confidence = result.get("confidence", 0)

            # Alert threshold: 60% — same as original CONFIDENCE_THRESHOLD
            if predicted in ("Cough", "Sneeze") and confidence >= 60.0:
                await sio_server.emit("alert", {
                    "type":       predicted,
                    "confidence": round(confidence, 2),
                    "timestamp":  result.get("timestamp"),
                })

                # DB logging at 60%+ to match the frontend alert threshold
                if confidence >= 60.0:
                    await _log_event_to_db(sid, predicted, confidence)

    except Exception as e:
        logger.error(f"Error processing audio chunk from {sid}: {e}")


async def _log_event_to_db(sid, event_type_str, confidence):
    """Log a confirmed detection to the database."""
    session = await sio_server.get_session(sid)
    if not session or not session.get("authenticated"):
        return

    type_map = {
        "Cough":   AudioEventType.cough,
        "Sneeze":  AudioEventType.sneeze,
        "Talking": AudioEventType.talking,
        "Noise":   AudioEventType.noise,
    }
    db_type = type_map.get(event_type_str)
    if not db_type:
        return

    db = SessionLocal()
    try:
        from tables.users import CareTaker
        username  = session.get("username")
        caretaker = db.query(CareTaker).filter(CareTaker.username == username).first()

        if caretaker:
            event = AudioEvent(
                caretaker_id      = caretaker.id,
                care_recipient_id = session.get("care_recipient_id"),
                event_type        = db_type,
                confidence        = confidence,
                duration_ms       = 500,
            )
            db.add(event)
            db.commit()
            logger.info(f"  [DB] Logged {event_type_str} ({confidence:.1f}%) for {username}")
        else:
            logger.warning(f"  [DB] CareTaker '{username}' not found. Aborting log.")
    except Exception as e:
        logger.error(f"DB logging failed: {e}")
    finally:
        db.close()
