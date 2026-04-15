import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SessionLocal
from tables.users import CareRecipient
from services.recommendation_engine import run_recommendation_engine

# Import ALL models to avoid SQLAlchemy mapper errors
from tables.medical_reports import MedicalReport
from tables.medical_conditions import PatientCondition, LabValue, MedicalAlert, LabOrderDetail
from tables.vital_signs import VitalSign
from tables.medications import Medication
from tables.video_analysis import VideoAnalysis
from tables.allergies import Allergy
from tables.conversation_history import ConversationMessage
from tables.environment import EnvironmentSensor

def main():
    db = SessionLocal()
    try:
        # 1. Ensure doctor email is set for demo
        recipient = db.query(CareRecipient).filter(CareRecipient.id == 8).first()
        if recipient:
            recipient.doctor_email = "doctor@caretaker.ai"
            db.commit()
            print(f"Set doctor_email for {recipient.full_name}")
        
        # 2. Trigger recommendation engine
        print("\nTriggering Recommendation Engine...")
        rec = run_recommendation_engine(8, db, trigger_type="demo_proactive_check")
        
        if rec:
            print("\n" + "="*50)
            print(f"PROACTIVE TREND SUMMARY:\n{rec.trend_summary}")
            print("\nSUGGESTIONS:")
            import json
            print(json.dumps(rec.suggestions_json, indent=2))
            
            # 3. Verify Alert was created
            alert = db.query(MedicalAlert).filter(
                MedicalAlert.care_recipient_id == 8
            ).order_by(MedicalAlert.created_at.desc()).first()
            if alert:
                print("\n" + "="*50)
                print(f"LATEST SYSTEM ALERT (Severity: {alert.severity.value}):")
                print(f"Message: {alert.message}")
        else:
            print("No new recommendations generated.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()
