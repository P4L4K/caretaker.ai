
from config import SessionLocal
from tables.vital_signs import VitalSign
from tables.users import CareRecipient
import tables.video_analysis # Register VideoAnalysis
import tables.medical_reports # Register MedicalReport
from tables.vital_signs import VitalSign
from tables.users import CareRecipient

db = SessionLocal()
try:
    count = db.query(VitalSign).count()
    print(f"Vital Signs count: {count}")
    
    first = db.query(VitalSign).first()
    if first:
        print(f"Sample: ID={first.id}, HR={first.heart_rate}, RecipientID={first.care_recipient_id}")
finally:
    db.close()
