"""Recommendation Engine — Actionable clinical intelligence.

Evaluates longitudinal lab history against deterministic clinical rules.
Produces safe, structured recommendations including lifestyle, diet, and clinical actions.
"""

from typing import List, Dict, Any, Optional
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.medical_conditions import LabValue
from tables.medical_recommendations import MedicalRecommendation

# ─── Clinical Rules ─────────────────────────────────────────────────────────
# Each rule maps an actionable condition to severity and concrete actions.

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
    {
        "metric": "Vitamin D",
        "thresholds": [
            {"max": 10, "severity": "high", "message": "Severe Vitamin D deficiency.",
             "actions": [{"type": "doctor_visit", "text": "Consult physician for prescription-strength Vitamin D therapy."}]},
            {"max": 20, "severity": "medium", "message": "Vitamin D deficiency.",
             "actions": [{"type": "diet", "text": "Consider OTC Vitamin D supplementation (consult pharmacist)."}, {"type": "lifestyle", "text": "Increase safe sunlight exposure."}]},
            {"max": 30, "severity": "low", "message": "Vitamin D insufficiency.",
             "actions": [{"type": "diet", "text": "Increase dietary intake (fortified dairy, fatty fish, eggs)."}]},
        ]
    },
    {
        "metric": "Vitamin B12",
        "thresholds": [
            {"max": 150, "severity": "high", "message": "Severe Vitamin B12 deficiency (Risk of neuropathy/anemia).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician for potential B12 injections."}]},
            {"max": 300, "severity": "medium", "message": "Suboptimal Vitamin B12 levels.",
             "actions": [{"type": "diet", "text": "Increase intake of meat, dairy, or B12 fortified foods."}, {"type": "diet", "text": "Consider B12 supplementation if vegetarian/vegan."}]},
        ]
    },
    {
        "metric": "Creatinine",
        "thresholds": [
            {"min": 2.5, "severity": "critical", "message": "Creatinine is dangerously elevated indicating renal injury.",
             "actions": [{"type": "doctor_visit", "text": "Immediate nephrology consultation required."}]},
            {"min": 1.4, "severity": "high", "message": "Creatinine is elevated (Kidney function impaired).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician immediately."}, {"type": "diet", "text": "Avoid NSAIDs (like ibuprofen) and excessive protein."}]},
        ]
    },
    {
        "metric": "eGFR",
        "thresholds": [
            {"max": 30, "severity": "critical", "message": "eGFR is critically low (Severe kidney disease).",
             "actions": [{"type": "doctor_visit", "text": "Urgent nephrology evaluation."}]},
            {"max": 60, "severity": "high", "message": "eGFR indicates moderate kidney disease.",
             "actions": [{"type": "doctor_visit", "text": "Consult your doctor to review all medications."}, {"type": "diet", "text": "Limit sodium and consult about potassium intake."}]},
        ]
    },
    {
        "metric": "Hemoglobin",
        "thresholds": [
            {"max": 7.0, "severity": "critical", "message": "Critically low Hemoglobin (Severe Anemia).",
             "actions": [{"type": "doctor_visit", "text": "Immediate medical attention required! Go to ER or consult doctor NOW."}]},
            {"max": 10.0, "severity": "high", "message": "Low Hemoglobin indicating moderate anemia.",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor to investigate cause of anemia."}]},
            {"max": 12.0, "severity": "medium", "message": "Hemoglobin is slightly below normal.",
             "actions": [{"type": "diet", "text": "Ensure adequate dietary iron, Vitamin C, and B12."}]},
        ]
    },
    {
        "metric": "Platelets",
        "thresholds": [
            {"max": 50, "severity": "critical", "message": "Critically low platelet count. High risk of bleeding.",
             "actions": [{"type": "doctor_visit", "text": "Seek urgent medical attention avoiding any physical trauma."}]},
            {"max": 100, "severity": "high", "message": "Low platelet count (Thrombocytopenia).",
             "actions": [{"type": "doctor_visit", "text": "Consult physician immediately to determine root cause."}]},
            {"min": 600, "severity": "high", "message": "Elevated platelet count (Thrombocytosis).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor for further hematology workup."}]},
        ]
    },
    {
        "metric": "TSH",
        "thresholds": [
            {"min": 10.0, "severity": "high", "message": "TSH is significantly elevated (Hypothyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor for probable thyroid hormone replacement."}]},
            {"min": 5.0, "severity": "medium", "message": "TSH is mildly elevated (Subclinical hypothyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Monitor for symptoms like fatigue/weight gain and consult doctor."}]},
            {"max": 0.1, "severity": "high", "message": "TSH is highly suppressed (Hyperthyroidism).",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor. Avoid excess iodine."}]},
        ]
    },
    {
        "metric": "ALT",
        "thresholds": [
            {"min": 200, "severity": "critical", "message": "ALT is extremely elevated (Acute liver injury warning).",
             "actions": [{"type": "doctor_visit", "text": "Seek urgent medical care and hepatology review."}]},
            {"min": 60, "severity": "high", "message": "ALT is elevated indicating liver inflammation.",
             "actions": [{"type": "doctor_visit", "text": "Consult doctor."}, {"type": "lifestyle", "text": "Strictly avoid alcohol and unauthorized supplements."}]},
            {"min": 45, "severity": "medium", "message": "ALT is borderline high.",
             "actions": [{"type": "lifestyle", "text": "Limit alcohol consumption. Monitor weight."}]},
        ]
    },
    {
        "metric": "Systolic BP",
        "thresholds": [
            {"min": 180, "severity": "critical", "message": "Hypertensive Crisis detected.",
             "actions": [{"type": "doctor_visit", "text": "Seek emergency medical care immediately!"}]},
            {"min": 140, "severity": "high", "message": "Stage 2 Hypertension.",
             "actions": [
                 {"type": "doctor_visit", "text": "Consult doctor for medication management."},
                 {"type": "diet", "text": "Strictly restrict sodium intake."},
                 {"type": "home_remedy", "text": "Hibiscus tea and Garlic can provide additional support for BP reduction.", "evidence": "traditional", "confidence": "high"}
             ]},
            {"min": 130, "severity": "medium", "message": "Stage 1 Hypertension.",
             "actions": [
                 {"type": "lifestyle", "text": "Engage in regular aerobic exercise and manage stress."},
                 {"type": "home_remedy", "text": "Drink Hibiscus tea daily to help manage blood pressure.", "evidence": "traditional", "confidence": "medium"}
             ]},
        ]
    },
]

# ─── Combination Rules ──────────────────────────────────────────────────────

def evaluate_combinations(latest_labs: Dict[str, Dict]) -> List[Dict]:
    combos = []
    
    # Combo 1: Diabetic Dyslipidemia (High HbA1c + High LDL/Triglycerides)
    hba1c = latest_labs.get("HbA1c")
    ldl = latest_labs.get("LDL")
    
    if hba1c and ldl and hba1c["value"] > 6.5 and ldl["value"] > 130:
        combos.append({
            "metric": "Diabetic Dyslipidemia",
            "severity": "high",
            "message": "Combined high blood sugar and high LDL cholesterol significantly increases cardiovascular risk.",
            "actions": [
                {"type": "doctor_visit", "text": "Consult cardiologist/endocrinologist for aggressive lipid management."},
                {"type": "diet", "text": "Adopt a strict heart-healthy, low-carb diet."}
            ],
            "value": None,
            "reference": None,
            "source": "hybrid"
        })
        
    return combos

# ─── Next Test Recommender ──────────────────────────────────────────────────

def evaluate_missing_tests(history: Dict[str, List[Dict]], now: datetime.datetime) -> List[Dict]:
    missing = []
    
    # Rule 1: If diabetic (HbA1c > 6.5 historically), should test every 6 months.
    hba1labs = history.get("HbA1c", [])
    if any(lab["value"] > 6.5 for lab in hba1labs):
        # find most recent
        if hba1labs:
            latest = max(hba1labs, key=lambda x: x["date"])
            latest_date = latest["date"]
            if isinstance(latest_date, datetime.date) and not isinstance(latest_date, datetime.datetime):
                latest_date = datetime.datetime.combine(latest_date, datetime.time.min)
                
            days_since = (now - latest_date).days
            if days_since > 180:
                missing.append({
                    "metric": "HbA1c",
                    "severity": "suggestion",
                    "message": f"It has been {days_since // 30} months since your last HbA1c test. Diabetic guidelines recommend testing every 3-6 months.",
                    "actions": [{"type": "test", "text": "Schedule an HbA1c test."}],
                    "value": None,
                    "reference": None,
                    "source": "rule"
                })
    
    return missing

# ─── Trend Detection ────────────────────────────────────────────────────────

def detect_trend(metric: str, labs: List[Dict]) -> Optional[str]:
    """Analyzes last 90 days of labs to detect worsening trends."""
    if len(labs) < 2:
        return None
        
    # Sort chronologically
    sorted_labs = sorted(labs, key=lambda x: x["date"])
    recent = [l for l in sorted_labs if (datetime.datetime.utcnow() - l["date"]).days <= 90]
    
    if len(recent) < 3:
        return None
        
    # Look at last 3 recent values
    last3 = recent[-3:]
    v1, v2, v3 = last3[0]["value"], last3[1]["value"], last3[2]["value"]
    
    # ── Trend Smoothing (Architect Rule #4) ──────────
    # Ignore tiny fluctuations (noise) < 5%
    avg = (abs(v1) + abs(v2) + abs(v3)) / 3
    if avg > 0:
        d1 = abs(v2 - v1) / avg
        d2 = abs(v3 - v2) / avg
        if d1 < 0.03 and d2 < 0.03: # Tight smoothing for individual steps
            return None
        # Overall trend delta
        total_delta = abs(v3 - v1) / avg
        if total_delta < 0.05:
            return None

    # "Up is bad" metrics
    if metric in ["HbA1c", "LDL", "Fasting Glucose", "Creatinine", "ALT", "TSH", "Systolic BP"]:
        if v3 > v2 and v2 > v1:
            return "increasing_bad"
            
    # "Down is bad" metrics
    if metric in ["eGFR", "HDL", "Vitamin D", "Vitamin B12", "Hemoglobin"]:
        if v3 < v2 and v2 < v1:
            return "decreasing_bad"
            
    return None

# ─── Safety Filter ──────────────────────────────────────────────────────────

BLOCKED_WORDS = ["prescribe", "dose", "dosage"]

def safety_filter(alerts: List[Dict]) -> List[Dict]:
    safe_alerts = []
    for a in alerts:
        msg = a["message"].lower()
        if any(w in msg for w in BLOCKED_WORDS):
            continue  # Drop unsafe system generation
            
        # Architect Rule #1: Sanitize medicine suggestions
        # Architect Rule #3: Hide home remedies in critical cases
        if a["severity"] == "critical":
            a["actions"] = [act for act in a["actions"] if act["type"] != "home_remedy"]
        
        # Guarantee safety disclaimer
        if not a["message"].endswith("."):
            a["message"] += "."
        
        # Proactively add 'Consult Doctor' if not present
        has_advice = any("doctor" in str(act).lower() or act["type"] == "doctor_visit" for act in a["actions"])
        if not has_advice:
            a["actions"].append({"type": "doctor_visit", "text": "Consult a doctor for clinical correlation."})
            
        a["message"] += " Please consult a qualified doctor before making health changes or taking new medications."
        
        safe_alerts.append(a)
    return safe_alerts

# ─── Prioritization & Deduplication ─────────────────────────────────────────

PRIORITY = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "suggestion": 1,
    "low": 0
}

def prioritize_and_dedup(alerts: List[Dict], db: Session, recipient_id: int, now: datetime.datetime) -> List[Dict]:
    # 1. Intra-run deduplication (keep highest severity per metric)
    best_alerts = {}
    for a in alerts:
        metric = a["metric"]
        if metric not in best_alerts:
            best_alerts[metric] = a
        else:
            if PRIORITY[a["severity"]] > PRIORITY[best_alerts[metric]["severity"]]:
                best_alerts[metric] = a
                
    unique_alerts = list(best_alerts.values())
    
    # 2. Database 24hr deduping
    final_alerts = []
    for a in unique_alerts:
        # Check if identical metric & severity emitted in last 24h
        recent = db.query(MedicalRecommendation).filter(
            MedicalRecommendation.care_recipient_id == recipient_id,
            MedicalRecommendation.metric == a["metric"],
            MedicalRecommendation.severity == a["severity"],
            MedicalRecommendation.created_at >= now - datetime.timedelta(days=1)
        ).first()
        
        if not recent:
            final_alerts.append(a)
            
    # 3. Sort by priority
    final_alerts.sort(key=lambda x: PRIORITY.get(x["severity"], 0), reverse=True)
    
    # 4. Enforce UI limit cap (Max 2 criticals, max 5 total - Architect Rule #10)
    criticals = [a for a in final_alerts if a["severity"] == "critical"][:2]
    others = [a for a in final_alerts if a["severity"] != "critical"]
    
    capped = (criticals + others)[:5]
    
    return capped

# ─── State of Health Engine ──────────────────────────────────────────────────

def get_state_of_health(recipient_id: int, db: Session) -> Dict[str, Any]:
    """Computes a high-level health category (Architect Rule #11)."""
    # Fetch recent recommendations
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id
    ).order_by(desc(MedicalRecommendation.created_at)).limit(10).all()
    
    if not recs:
        return {"category": "Stable", "color": "var(--success)", "icon": "fa-check-circle"}
        
    highest_sev = "low"
    for r in recs:
        if PRIORITY[r.severity] > PRIORITY[highest_sev]:
            highest_sev = r.severity
            
    if highest_sev == "critical":
        return {"category": "Critical Risk", "color": "var(--danger)", "icon": "fa-exclamation-triangle", "label": "Immediate attention required"}
    if highest_sev == "high":
        return {"category": "High Concern", "color": "#f97316", "icon": "fa-bolt", "label": "Rising health risks detected"}
    if highest_sev == "medium":
        return {"category": "Moderate Risk", "color": "var(--warning)", "icon": "fa-info-circle", "label": "Health monitoring advised"}
        
    return {"category": "Good", "color": "var(--success)", "icon": "fa-check-circle", "label": "Maintain current lifestyle"}

# ─── Main Execution Pipeline ──────────────────────────────────────────────────

def generate_recommendations(recipient_id: int, db: Session):
    now = datetime.datetime.utcnow()
    
    # 1. Fetch entire lab history
    all_labs = db.query(LabValue).filter(LabValue.care_recipient_id == recipient_id).all()
    
    # Group by metric
    history = {}
    latest_labs = {}
    
    for lab in all_labs:
        name = lab.metric_name
        if name not in history:
            history[name] = []
        # Normalize date to datetime for calculations
        recorded_at = lab.recorded_date or lab.created_at
        if isinstance(recorded_at, datetime.date) and not isinstance(recorded_at, datetime.datetime):
            recorded_at = datetime.datetime.combine(recorded_at, datetime.time.min)

        entry = {
            "value": lab.normalized_value or lab.metric_value,
            "unit": lab.normalized_unit or lab.unit,
            "date": recorded_at,
            "ref": f"{lab.reference_range_low or ''}-{lab.reference_range_high or ''}"
        }
        history[name].append(entry)
        
        if name not in latest_labs or entry["date"] >= latest_labs[name]["date"]:
            latest_labs[name] = entry

    alerts = []

    # 2. Evaluate Individual Rules (against LATEST lab)
    for name, latest in latest_labs.items():
        val = latest["value"]
        
        for rule in RULES:
            if rule["metric"] == name:
                matched = False
                for t in rule["thresholds"]:
                    hit = False
                    if "min" in t and val >= t["min"]: hit = True
                    if "max" in t and val <= t["max"]: hit = True
                    
                    if hit:
                        alert = {
                            "metric": name,
                            "severity": t["severity"],
                            "message": t["message"],
                            "actions": t["actions"],
                            "value": val,
                            "reference": latest["ref"],
                            "source": "rule"
                        }
                        # Trend inspection
                        trend = detect_trend(name, history[name])
                        if trend == "increasing_bad":
                            alert["message"] += " Condition is worsening over recent reports."
                            alert["source"] = "trend"
                        elif trend == "decreasing_bad":
                            alert["message"] += " Condition is deteriorating over recent reports."
                            alert["source"] = "trend"
                            
                        alerts.append(alert)
                        matched = True
                        break # Only trigger highest matching severity per rule
                        
                if matched:
                    break

    # 3. Evaluate Combo Rules
    alerts.extend(evaluate_combinations(latest_labs))
    
    # 4. Evaluate Missing Tests
    alerts.extend(evaluate_missing_tests(history, now))

    # 5. Safety Filter
    safe_alerts = safety_filter(alerts)
    
    # 6. Prioritize & Dedup
    final_alerts = prioritize_and_dedup(safe_alerts, db, recipient_id, now)
    
    # 7. Save to DB
    for a in final_alerts:
        rec = MedicalRecommendation(
            care_recipient_id=recipient_id,
            metric=a["metric"],
            severity=a["severity"],
            message=a["message"],
            trigger_value=a["value"],
            reference_range=a.get("reference"),
            source=a.get("source", "rule"),
            confidence_score=1.0, # Rule-based deterministic
            actions=a["actions"],
            created_at=now
        )
        db.add(rec)
        
    db.commit()
    print(f"[recommendation_engine] Generated {len(final_alerts)} validated clinical recommendations.")
