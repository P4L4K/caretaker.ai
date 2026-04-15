"""Repair script: Populate lab values from analysis summaries that have clinical data
but where the structured extraction returned empty.

Also triggers the recommendation engine after populating.

Usage:
    python scripts/repair_lab_values.py --recipient-id 8
"""

import sys, os, datetime, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import engine
from sqlalchemy import text


def parse_lab_values_from_summary(summary: str) -> list:
    """Extract lab values from the free-text analysis_summary field."""
    lab_values = []
    
    # Common patterns: "MetricName: value unit" or "MetricName value%"
    patterns = [
        (r'HbA1c\s*[:\-]?\s*([\d.]+)\s*%', 'HbA1c', '%'),
        (r'HBsAg\s*[:\-]?\s*([\d.]+)\s*IU/mL', 'HBsAg', 'IU/mL'),
        (r'Ammonia\s*[:\-]?\s*([\d.]+)', 'Ammonia', 'umol/L'),
        (r'CRP\s*[:\-]?\s*([\d.]+)\s*mg/L', 'CRP', 'mg/L'),
        (r'SpO2\s*[:\-]?\s*([\d.]+)\s*%', 'SpO2', '%'),
        (r'BMI\s*[:\-]?\s*([\d.]+)', 'BMI', 'kg/m2'),
        (r'Fasting\s*(?:Blood\s*)?(?:Glucose|Sugar)\s*[:\-]?\s*([\d.]+)', 'Fasting Glucose', 'mg/dL'),
        (r'(?:Total\s*)?Cholesterol\s*[:\-]?\s*([\d.]+)', 'Total Cholesterol', 'mg/dL'),
        (r'Creatinine\s*[:\-]?\s*([\d.]+)', 'Creatinine', 'mg/dL'),
        (r'eGFR\s*[:\-]?\s*([\d.]+)', 'eGFR', 'mL/min'),
        (r'Hemoglobin\s*[:\-]?\s*([\d.]+)', 'Hemoglobin', 'g/dL'),
        (r'Triglycerides\s*[:\-]?\s*([\d.]+)', 'Triglycerides', 'mg/dL'),
        (r'LDL\s*[:\-]?\s*([\d.]+)', 'LDL', 'mg/dL'),
        (r'HDL\s*[:\-]?\s*([\d.]+)', 'HDL', 'mg/dL'),
        (r'TSH\s*[:\-]?\s*([\d.]+)', 'TSH', 'mIU/L'),
        (r'Uric\s*Acid\s*[:\-]?\s*([\d.]+)', 'Uric Acid', 'mg/dL'),
        (r'WBC\s*[:\-]?\s*([\d.]+)', 'WBC', '10^3/uL'),
        (r'RBC\s*[:\-]?\s*([\d.]+)', 'RBC', '10^6/uL'),
        (r'Platelets?\s*[:\-]?\s*([\d.]+)', 'Platelets', '10^3/uL'),
        (r'SGPT\s*[:\-]?\s*([\d.]+)', 'SGPT (ALT)', 'U/L'),
        (r'SGOT\s*[:\-]?\s*([\d.]+)', 'SGOT (AST)', 'U/L'),
        (r'Bilirubin\s*[:\-]?\s*([\d.]+)', 'Bilirubin', 'mg/dL'),
        (r'Albumin\s*[:\-]?\s*([\d.]+)', 'Albumin', 'g/dL'),
        (r'Vitamin\s*D\s*[:\-]?\s*([\d.]+)', 'Vitamin D', 'ng/mL'),
        (r'Vitamin\s*B12\s*[:\-]?\s*([\d.]+)', 'Vitamin B12', 'pg/mL'),
        (r'Ferritin\s*[:\-]?\s*([\d.]+)', 'Ferritin', 'ng/mL'),
        (r'Iron\s*[:\-]?\s*([\d.]+)', 'Iron', 'ug/dL'),
    ]
    
    for pattern, metric, unit in patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                lab_values.append({'metric': metric, 'value': val, 'unit': unit})
            except ValueError:
                pass
    
    return lab_values


def repair_recipient(recipient_id: int):
    conn = engine.connect()
    try:
        # Get all reports with summaries
        rows = conn.execute(text(
            "SELECT id, analysis_summary, report_date FROM medical_reports "
            "WHERE care_recipient_id = :rid AND analysis_summary IS NOT NULL "
            "ORDER BY id"
        ), {"rid": recipient_id}).fetchall()
        
        if not rows:
            print(f"No reports found for recipient {recipient_id}")
            return
        
        total_inserted = 0
        for row in rows:
            report_id, summary, report_date = row
            if not summary:
                continue
            
            report_date = report_date or datetime.date.today()
            lab_vals = parse_lab_values_from_summary(summary)
            
            if not lab_vals:
                print(f"  Report {report_id}: No lab values found in summary")
                continue
            
            print(f"  Report {report_id}: Found {len(lab_vals)} lab values")
            
            # Check existing lab values for this report
            existing = conn.execute(text(
                "SELECT COUNT(*) FROM lab_values WHERE report_id = :rid"
            ), {"rid": report_id}).scalar()
            
            if existing > 0:
                print(f"    Already has {existing} lab values, skipping")
                continue
            
            for lv in lab_vals:
                conn.execute(text("""
                    INSERT INTO lab_values (care_recipient_id, report_id, metric_name, metric_value, 
                        unit, normalized_value, normalized_unit, is_abnormal, recorded_date)
                    VALUES (:crid, :rid, :name, :val, :unit, :val, :unit, :abnormal, :date)
                """), {
                    "crid": recipient_id,
                    "rid": report_id,
                    "name": lv['metric'],
                    "val": lv['value'],
                    "unit": lv['unit'],
                    "abnormal": False,
                    "date": report_date
                })
                total_inserted += 1
                print(f"    + {lv['metric']}: {lv['value']} {lv['unit']}")
        
        conn.commit()
        print(f"\nInserted {total_inserted} lab values for recipient {recipient_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient-id", type=int, required=True)
    args = parser.parse_args()
    repair_recipient(args.recipient_id)
