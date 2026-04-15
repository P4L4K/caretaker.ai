import os
import sys
import sqlalchemy
from datetime import datetime

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import engine, SessionLocal

def cleanup():
    print("--- Caretaker.AI Junk Alert Cleanup ---")
    session = SessionLocal()
    try:
        # Find Janak Devi
        res = session.execute(sqlalchemy.text("SELECT id, full_name FROM care_recipients WHERE full_name ILIKE '%Janak%'")).fetchone()
        if not res:
            print("Janak Devi not found in DB.")
            return
        
        rid = res[0]
        name = res[1]
        print(f"Target Recipient: {name} (ID: {rid})")

        # 1. Purge junk VideoAnalysis records
        res = session.execute(sqlalchemy.text(
            "DELETE FROM video_analysis "
            "WHERE recipient_id = :rid AND fall_count > 10"
        ), {"rid": rid})
        
        print(f"Purged {res.rowcount} junk VideoAnalysis records.")

        session.commit()
        print("Cleanup Complete.")

    except Exception as e:
        session.rollback()
        print(f"Cleanup Failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    cleanup()
