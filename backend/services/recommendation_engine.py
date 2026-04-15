"""Hybrid Recommendation Engine — Actionable clinical intelligence.

Combines deterministic clinical rules with proactive AI-driven trend analysis
to produce safe, structured, and personalized health recommendations.
"""

import json
import re
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.medical_conditions import LabValue
from tables.medical_recommendations import MedicalRecommendation
from tables.health_recommendations import HealthRecommendation
from utils.gemini_client import call_gemini

# ─── Clinical Rules (Deterministic) ─────────────────────────────────────────

RULES = [
    {
        "metric": "HbA1c",
        "thresholds": [
            {"min": 8.0, "severity": "critical", "message": "Blood sugar control is significantly impaired.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult physician for medication review immediately."},
                 {"type": "test", "text": "Monitor daily blood glucose fasting and PP."}
             ]},
            {"min": 6.5, "severity": "high", "message": "Blood sugar levels indicate diabetes.",
             "actions": [
                 {"type": "doctor_visit", "text": "Discuss diabetes management plan with your doctor."},
                 {"type": "diet", "text": "Strictly reduce refined sugars and carbohydrates."},
                 {"type": "lifestyle", "text": "Increase aerobic physical activity."},
                 {"type": "home_remedy", "text": "Fenugreek (Methi) can complement medical treatment for sugar control.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 5.7, "severity": "medium", "message": "Blood sugar levels indicate prediabetes risk.",
             "actions": [
                 {"type": "diet", "text": "Reduce intake of high glycemic index foods."},
                 {"type": "lifestyle", "text": "Maintain a healthy weight."},
                 {"type": "home_remedy", "text": "Consume Fenugreek (Methi) seeds soaked in water to help lower blood sugar.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "Fasting Glucose",
        "thresholds": [
            {"min": 200, "severity": "critical", "message": "Extremely high fasting blood glucose.",
             "actions": [{"type": "doctor_visit", "text": "Seek immediate medical attention."}]},
            {"min": 126, "severity": "high", "message": "Fasting blood sugar is in the diabetic range.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult doctor for formal diagnosis."},
                 {"type": "home_remedy", "text": "Incorporate Cinnamon and Fenugreek into your daily routine.", "evidence": "traditional", "confidence": "low"}
             ]},
            {"min": 100, "severity": "medium", "message": "Fasting blood sugar is mildly elevated.",
             "actions": [
                 {"type": "diet", "text": "Monitor simple carbohydrate intake."},
                 {"type": "home_remedy", "text": "A pinch of Cinnamon in warm water can improve insulin sensitivity.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "LDL",
        "thresholds": [
            {"min": 190, "severity": "critical", "message": "LDL cholesterol is dangerously high.",
             "actions": [{"type": "doctor_visit", "text": "Consult cardiologist. Statin therapy may be required."}]},
            {"min": 160, "severity": "high", "message": "LDL cholesterol is very high.",
             "actions": [
                 {"type": "diet", "text": "Significantly reduce saturated and trans fats."},
                 {"type": "doctor_visit", "text": "Discuss lipid-lowering options with your doctor."},
                 {"type": "home_remedy", "text": "Garlic and Oats are highly effective at this stage for lipid management.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 130, "severity": "medium", "message": "LDL cholesterol is borderline high.",
             "actions": [
                 {"type": "diet", "text": "Increase soluble dietary fiber (oats, beans)."},
                 {"type": "lifestyle", "text": "Incorporate regular cardio exercise."},
                 {"type": "home_remedy", "text": "Consume 1 clove of raw Garlic daily to naturally lower cholesterol.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
    {
        "metric": "HDL",
        "thresholds": [
            {"max": 35, "severity": "high", "message": "HDL (good) cholesterol is very low.",
             "actions": [{"type": "lifestyle", "text": "Quit smoking if applicable."}, {"type": "diet", "text": "Consume omega-3 rich foods (fish, walnuts)."}]},
            {"max": 40, "severity": "medium", "message": "HDL (good) cholesterol is below optimal.",
             "actions": [{"type": "lifestyle", "text": "Increase moderate aerobic exercise to boost HDL."}]},
        ]
    },
    {
        "metric": "Triglycerides",
        "thresholds": [
            {"min": 500, "severity": "critical", "message": "Triglycerides are extremely high (Pancreatitis risk).",
             "actions": [{"type": "doctor_visit", "text": "Seek immediate medical attention for triglyceride reduction."}]},
            {"min": 200, "severity": "high", "message": "Triglycerides are high.",
             "actions": [{"type": "diet", "text": "Eliminate alcohol and refined sugars."}, {"type": "test", "text": "Re-test lipid profile in 3 months."}]},
            {"min": 150, "severity": "medium", "message": "Triglycerides are borderline high.",
             "actions": [{"type": "diet", "text": "Limit sugary drinks and excess carbohydrates."}]},
        ]
    },
]

def detect_trend(metric: str, labs: List[Dict]) -> Optional[str]:
    if len(labs) < 2: return None
    sorted_labs = sorted(labs, key=lambda x: x["date"])
    recent = [l for l in sorted_labs if (datetime.datetime.utcnow() - l["date"]).days <= 90]
    if len(recent) < 3: return None
    last3 = recent[-3:]; v1, v2, v3 = last3[0]["value"], last3[1]["value"], last3[2]["value"]
    avg = (abs(v1) + abs(v2) + abs(v3)) / 3
    if avg > 0:
        if abs(v2 - v1)/avg < 0.03 and abs(v3 - v2)/avg < 0.03: return None
        if abs(v3 - v1)/avg < 0.05: return None
    if v3 > v2 and v2 > v1: return "increasing"
    if v3 < v2 and v2 < v1: return "decreasing"
    return None

def safety_filter(alerts: List[Dict]) -> List[Dict]:
    safe = []
    for a in alerts:
        msg = a["message"].lower()
        if any(w in msg for w in ["prescribe", "dose"]): continue
        if not a["message"].endswith("."): a["message"] += "."
        a["message"] += " Consult a doctor before making changes."
        safe.append(a)
    return safe

# ─── Main Logic ─────────────────────────────────────────────────────────────

def run_recommendation_engine(recipient_id: int, db: Session, trigger_type: str = "report"):
    """Unified engine: Runs deterministic rules AND AI Trend analysis."""
    now = datetime.datetime.utcnow()
    from tables.users import CareRecipient
    from tables.medical_conditions import PatientCondition, ConditionStatus, LabValue
    from tables.vital_signs import VitalSign
    from tables.medications import Medication, MedicationStatus

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient: return

    # 1. Deterministic Rule Phase (Saves to MedicalRecommendation)
    all_labs = db.query(LabValue).filter(LabValue.care_recipient_id == recipient_id).all()
    latest_labs = {}
    for lab in all_labs:
        name = lab.metric_name
        d = lab.recorded_date or lab.created_at
        if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime): d = datetime.datetime.combine(d, datetime.time.min)
        if name not in latest_labs or d >= latest_labs[name]["date"]:
            latest_labs[name] = {"value": lab.normalized_value or lab.metric_value, "date": d, "ref": f"{lab.reference_range_low or ''}-{lab.reference_range_high or ''}"}

    rule_alerts = []
    for name, latest in latest_labs.items():
        val = latest["value"]
        for rule in RULES:
            if rule["metric"] == name:
                for t in rule["thresholds"]:
                    if ("min" in t and val >= t["min"]) or ("max" in t and val <= t["max"]):
                        rule_alerts.append({"metric": name, "severity": t["severity"], "message": t["message"], "actions": t["actions"], "value": val, "reference": latest["ref"]})
                        break

    for a in safety_filter(rule_alerts):
        db.add(MedicalRecommendation(care_recipient_id=recipient_id, metric=a["metric"], severity=a["severity"], message=a["message"], trigger_value=a["value"], reference_range=a["reference"], actions=a["actions"], created_at=now))

    # 2. AI Trend Phase (Saves to HealthRecommendation)
    try:
        lab_hist = [{"metric_name": l.metric_name, "normalized_value": l.normalized_value, "recorded_date": l.recorded_date} for l in all_labs]
        vitals_raw = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id).order_by(VitalSign.recorded_at.desc()).limit(20).all()
        if len(lab_hist) >= 2 or len(vitals_raw) >= 2:
            prompt = f"Analyze health trends for {recipient.full_name}. Lab hist: {lab_hist[:10]}. Vitals: {[v.heart_rate for v in vitals_raw[:5]]}. Return JSON with [trend_summary, diet, lifestyle, medication_suggestions, next_tests]."
            ai_data = call_gemini({"contents": [{"parts": [{"text": prompt}]}]}, timeout=45)
            if ai_data and "candidates" in ai_data:
                res = ai_data["candidates"][0]["content"]["parts"][0]["text"]
                # Clean and save to HealthRecommendation
                cleaned = re.sub(r"```json|```", "", res).strip()
                parsed = json.loads(cleaned)
                db.add(HealthRecommendation(care_recipient_id=recipient_id, trigger_type=trigger_type, trend_summary=parsed.get("trend_summary"), suggestions_json=parsed, generated_at=now))
    except Exception as e:
        print(f"[recommendation_engine] AI path failed: {e}")

    db.commit()
    print(f"[recommendation_engine] Hybrid run completed for recipient {recipient_id}")

def generate_recommendations(recipient_id: int, db: Session):
    """Alias for backward compatibility."""
    run_recommendation_engine(recipient_id, db)
