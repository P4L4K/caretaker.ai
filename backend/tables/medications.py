from sqlalchemy import Column, Integer, String, Date, ForeignKey, Enum
from sqlalchemy.orm import relationship
from config import Base
import enum

class MedicationStatus(str, enum.Enum):
    active = "active"
    completed = "completed"
    terminated = "terminated"

class Medication(Base):
    __tablename__ = "medications"

    medication_id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="CASCADE"), nullable=False, index=True)
    
    medicine_name = Column(String, nullable=False)
    dosage = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    schedule_time = Column(String, nullable=True)
    
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    status = Column(Enum(MedicationStatus), default=MedicationStatus.active, nullable=False)

    care_recipient = relationship("CareRecipient", back_populates="active_medications")

class MedicationHistory(Base):
    __tablename__ = "medication_history"

    history_id = Column(Integer, primary_key=True, index=True)
    care_recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="CASCADE"), nullable=False, index=True)
    
    medicine_name = Column(String, nullable=False)
    dosage = Column(String, nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    termination_reason = Column(String, nullable=True)

    care_recipient = relationship("CareRecipient", back_populates="medication_history")
