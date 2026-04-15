"""Seed Demo Vitals -- Populates 7 days of realistic vitals data for demo/evaluation.

Usage:
    cd backend
    python scripts/seed_demo_vitals.py --recipient-id 8
"""

import sys
import os
import datetime
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import engine
from sqlalchemy import text


# 7-day realistic vitals for a T2DM + Hypertension patient
DEMO_VITALS = [
    {"day_offset": -6, "heart_rate": 88, "systolic_bp": 148, "diastolic_bp": 92, "oxygen_saturation": 96, "temperature": 37.1, "sleep_score": 6, "weight": 72.0},
    {"day_offset": -5, "heart_rate": 85, "systolic_bp": 145, "diastolic_bp": 90, "oxygen_saturation": 97, "temperature": 37.0, "sleep_score": 7, "weight": 71.8},
    {"day_offset": -4, "heart_rate": 91, "systolic_bp": 152, "diastolic_bp": 95, "oxygen_saturation": 95, "temperature": 37.3, "sleep_score": 5, "weight": 72.1},
    {"day_offset": -3, "heart_rate": 83, "systolic_bp": 143, "diastolic_bp": 88, "oxygen_saturation": 97, "temperature": 36.9, "sleep_score": 7, "weight": 71.5},
    {"day_offset": -2, "heart_rate": 86, "systolic_bp": 147, "diastolic_bp": 91, "oxygen_saturation": 96, "temperature": 37.1, "sleep_score": 6, "weight": 71.7},
    {"day_offset": -1, "heart_rate": 89, "systolic_bp": 150, "diastolic_bp": 93, "oxygen_saturation": 95, "temperature": 37.2, "sleep_score": 6, "weight": 72.3},
    {"day_offset":  0, "heart_rate": 82, "systolic_bp": 141, "diastolic_bp": 87, "oxygen_saturation": 98, "temperature": 36.8, "sleep_score": 8, "weight": 71.2},
]


def seed_vitals(recipient_id):
    conn = engine.connect()
    try:
        row = conn.execute(text("SELECT full_name, height, weight FROM care_recipients WHERE id = :id"), {"id": recipient_id}).fetchone()
        if not row:
            print(f"CareRecipient id={recipient_id} not found!")
            return False

        name, profile_height, profile_weight = row
        h = profile_height or 170
        print(f"Seeding 7 days of vitals for: {name} (id={recipient_id}, H={h}cm)")

        now = datetime.datetime.utcnow()
        count = 0

        for entry in DEMO_VITALS:
            target_date = now + datetime.timedelta(
                days=entry["day_offset"],
                hours=random.randint(-2, 2),
                minutes=random.randint(0, 59)
            )

            w = entry["weight"]
            height_m = h / 100.0
            bmi = round(w / (height_m * height_m), 1)

            conn.execute(text("""
                INSERT INTO vital_signs (care_recipient_id, heart_rate, systolic_bp, diastolic_bp,
                    oxygen_saturation, temperature, sleep_score, weight, height, bmi, recorded_at)
                VALUES (:rid, :hr, :sbp, :dbp, :o2, :temp, :sleep, :w, :h, :bmi, :ts)
            """), {
                "rid": recipient_id,
                "hr": entry["heart_rate"],
                "sbp": entry["systolic_bp"],
                "dbp": entry["diastolic_bp"],
                "o2": entry["oxygen_saturation"],
                "temp": entry["temperature"],
                "sleep": entry["sleep_score"],
                "w": w, "h": h, "bmi": bmi,
                "ts": target_date
            })
            count += 1
            print(f"  Day {entry['day_offset']:+d}: HR={entry['heart_rate']} BP={entry['systolic_bp']}/{entry['diastolic_bp']} SpO2={entry['oxygen_saturation']} BMI={bmi}")

        conn.commit()
        print(f"\nSeeded {count} vitals records successfully!")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed demo vitals data")
    parser.add_argument("--recipient-id", type=int, required=True)
    args = parser.parse_args()
    success = seed_vitals(args.recipient_id)
    sys.exit(0 if success else 1)
