"""Health Recommendations Table — Stores AI-generated proactive health suggestions.

Each recommendation is auto-generated after a new lab report is processed or
manual vitals are logged. The recommendation engine detects worsening trends
and generates structured, actionable suggestions.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from config import Base
import datetime


class HealthRecommendation(Base):
    __tablename__ = "health_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(
        Integer,
        ForeignKey("care_recipients.id", ondelete="CASCADE"),
        nullable=False, index=True
    )

    # What triggered this recommendation
    trigger_type = Column(String, nullable=False)  # "report" or "vitals"
    generated_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Content
    trend_summary = Column(Text, nullable=True)         # Human-readable trend overview
    suggestions_json = Column(JSON, nullable=True)      # Full structured suggestions
    # {
    #   "diet": [...],
    #   "lifestyle": [...],
    #   "medication_suggestions": [...],
    #   "urgent_flags": [...],
    #   "next_tests": [...]
    # }

    is_read = Column(Boolean, default=False)

    # Relationship
    care_recipient = relationship("CareRecipient")
