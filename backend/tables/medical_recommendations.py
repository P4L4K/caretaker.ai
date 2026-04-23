from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
import datetime

from config import Base

class MedicalRecommendation(Base):
    """
    Stores actionable clinical recommendations generated deterministically
    from the patient's lab value history.
    """
    __tablename__ = "medical_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="CASCADE"), nullable=False)
    
    # Recommendation identifiers
    metric = Column(String, nullable=False)  # Example: "HbA1c" or "Combination"
    severity = Column(String, nullable=False) # "critical", "high", "medium", "low", "suggestion"
    message = Column(Text, nullable=False)    # Explanatory text
    
    # Track the exact trigger (for traceability)
    trigger_value = Column(Float, nullable=True)
    reference_range = Column(String, nullable=True)
    
    # Ensure trust
    source = Column(String, default="rule")    # "rule", "trend", "hybrid"
    confidence_score = Column(Float, default=1.0)
    
    # List of grouped actions. Each action dict should have {"type": "doctor_visit", "text": "Consult doctor"}
    actions = Column(JSON, default=list)
    
    # Structured data for automated backend processing (e.g. specific drug names, test types)
    action_payload = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    recipient = relationship("CareRecipient", backref="recommendations")
