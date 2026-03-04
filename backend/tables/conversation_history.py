from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from config import Base
import datetime
import enum

class SenderEnum(str, enum.Enum):
    user = "user"
    bot = "bot"
    system = "system"

class MoodEnum(str, enum.Enum):
    happy = "happy"
    sad = "sad"
    anxious = "anxious"
    angry = "angry"
    neutral = "neutral"
    distressed = "distressed"

class TriggerTypeEnum(str, enum.Enum):
    user_initiated = "user_initiated"
    proactive_checkin = "proactive_checkin"
    reminder = "reminder"
    alert = "alert"
    scheduled = "scheduled"

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="CASCADE"), nullable=False)
    
    sender = Column(Enum(SenderEnum), nullable=False)
    message_text = Column(Text, nullable=False)
    mood_detected = Column(Enum(MoodEnum), nullable=True)
    mood_confidence = Column(Float, nullable=True)
    
    conversation_session_id = Column(String, index=True, nullable=False)
    trigger_type = Column(Enum(TriggerTypeEnum), nullable=False, default=TriggerTypeEnum.user_initiated)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    care_recipient = relationship("CareRecipient", back_populates="conversation_messages")


class ReminderTypeEnum(str, enum.Enum):
    water = "water"
    food = "food"
    medicine = "medicine"
    exercise = "exercise"
    custom = "custom"

class RecurrenceEnum(str, enum.Enum):
    once = "once"
    daily = "daily"
    custom = "custom"

class ProactiveReminder(Base):
    __tablename__ = "proactive_reminders"

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="CASCADE"), nullable=False)
    
    reminder_type = Column(Enum(ReminderTypeEnum), nullable=False)
    reminder_text = Column(String, nullable=False)
    scheduled_time = Column(String, nullable=False) # Store as "HH:MM"
    recurrence = Column(Enum(RecurrenceEnum), nullable=False, default=RecurrenceEnum.daily)
    
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    care_recipient = relationship("CareRecipient", back_populates="proactive_reminders")
