"""
Caretaker.ai — Sensor Fusion Insights Engine
=============================================
Aggregates ALL data sources (vitals, environment, medical reports, audio events,
video analysis) and generates cross-domain health conclusions using:
  1. Deterministic rule-based sensor fusion (always fast, always reproducible)
  2. Gemini AI interpretation (deep clinical insights, natural language)

Architecture:
  Body Sensors (ESP32) → Vitals (HR, BP, SpO2, Temp)
  Room Sensors         → Environment (Temp, Humidity, AQI)
  Camera               → Video Analysis (Falls, Activity, Mobility)
  Microphone           → Audio Events (Cough, Sneeze frequency)
  Medical Reports      → Conditions, Lab Values, Risk Score
       ↓ ALL MERGE HERE ↓
  [Sensor Fusion Engine] → Cross-Domain Conclusions → Dashboard + Alerts
"""

import os
import json
import datetime
import requests
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from tables.users import CareRecipient
from tables.vital_signs import VitalSign
from tables.audio_events import AudioEvent, AudioEventType
from tables.video_analysis import VideoAnalysis
from tables.medical_conditions import PatientCondition, LabValue, ConditionStatus


# ═══════════════════════════════════════════════════════════════════════
#  THRESHOLDS — Clinical reference ranges for sensor fusion rules
# ═══════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "heart_rate":         {"low": 60, "high": 100, "critical_low": 50, "critical_high": 130},
    "systolic_bp":        {"low": 90, "high": 130, "critical_low": 80, "critical_high": 180},
    "diastolic_bp":       {"low": 60, "high": 85, "critical_low": 50, "critical_high": 120},
    "oxygen_saturation":  {"low": 95, "critical_low": 90},
    "temperature_f":      {"low": 97.0, "high": 99.5, "critical_low": 95.0, "critical_high": 103.0},
    "sleep_score":        {"low": 50, "good": 70},
    "bmi":                {"underweight": 18.5, "normal_high": 24.9, "overweight": 30.0},
    "room_temp_c":        {"low": 18, "high": 26, "critical_low": 15, "critical_high": 32},
    "humidity":           {"low": 30, "high": 60, "critical_high": 80},
    "aqi_epa":            {"moderate": 2, "unhealthy": 4, "hazardous": 5},
    "cough_daily":        {"elevated": 5, "high": 15, "critical": 30},
    "fall_risk_days":     7,
}

# Severity Levels
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MODERATE = "moderate"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"


# ═══════════════════════════════════════════════════════════════════════
#  DATA AGGREGATOR — Collects all data for a recipient
# ═══════════════════════════════════════════════════════════════════════

def aggregate_patient_data(recipient_id: int, db: Session) -> dict:
    """
    Collect ALL available data for a recipient from every subsystem.
    This is the single source of truth for the insights engine.
    """
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {"error": "Recipient not found"}

    # 1. Latest Vitals (from ESP32 sensors)
    latest_vital = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id
    ).order_by(desc(VitalSign.recorded_at)).first()

    # Recent vitals for trend analysis (last 24h)
    vitals_24h = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient_id,
        VitalSign.recorded_at >= datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    ).order_by(desc(VitalSign.recorded_at)).all()

    # 2. Audio Events (cough/sneeze in last 7 days)
    audio_cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    cough_count_7d = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.event_type == AudioEventType.cough,
        AudioEvent.detected_at >= audio_cutoff
    ).scalar() or 0

    sneeze_count_7d = db.query(func.count(AudioEvent.id)).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.event_type == AudioEventType.sneeze,
        AudioEvent.detected_at >= audio_cutoff
    ).scalar() or 0

    # Daily average coughs
    cough_daily_avg = round(cough_count_7d / 7, 1)

    # 3. Video Analysis (falls, activity in last 7 days)
    video_cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=THRESHOLDS["fall_risk_days"])
    recent_videos = db.query(VideoAnalysis).filter(
        VideoAnalysis.recipient_id == recipient_id,
        VideoAnalysis.timestamp >= video_cutoff
    ).all()

    fall_count_7d = sum(v.fall_count for v in recent_videos)
    has_recent_fall = any(v.has_fall for v in recent_videos)
    avg_activity = round(sum(v.activity_score for v in recent_videos) / max(len(recent_videos), 1), 1)
    avg_mobility = round(sum(v.mobility_score for v in recent_videos) / max(len(recent_videos), 1), 1)

    # 4. Medical Conditions
    active_conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()

    # 5. Recent Lab Values
    recent_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id
    ).order_by(desc(LabValue.recorded_date)).limit(20).all()

    abnormal_labs = [l for l in recent_labs if l.is_abnormal]

    # 6. Environmental Data (from weather API — fetched separately, passed as param or fetched live)
    env_data = _fetch_environment_data(recipient)

    return {
        "recipient": {
            "id": recipient.id,
            "name": recipient.full_name,
            "age": recipient.age,
            "gender": recipient.gender.value if recipient.gender else None,
            "respiratory_condition": recipient.respiratory_condition_status,
            "risk_score": recipient.risk_score,
        },
        "vitals": {
            "heart_rate": latest_vital.heart_rate if latest_vital else None,
            "systolic_bp": latest_vital.systolic_bp if latest_vital else None,
            "diastolic_bp": latest_vital.diastolic_bp if latest_vital else None,
            "oxygen_saturation": latest_vital.oxygen_saturation if latest_vital else None,
            "temperature": latest_vital.temperature if latest_vital else None,
            "sleep_score": latest_vital.sleep_score if latest_vital else None,
            "bmi": latest_vital.bmi if latest_vital else None,
            "recorded_at": latest_vital.recorded_at.isoformat() if latest_vital and latest_vital.recorded_at else None,
            "readings_24h": len(vitals_24h),
        },
        "audio": {
            "cough_count_7d": cough_count_7d,
            "sneeze_count_7d": sneeze_count_7d,
            "cough_daily_avg": cough_daily_avg,
        },
        "video": {
            "fall_count_7d": fall_count_7d,
            "has_recent_fall": has_recent_fall,
            "avg_activity_score": avg_activity,
            "avg_mobility_score": avg_mobility,
        },
        "medical": {
            "active_conditions": [{"name": c.disease_name, "status": c.status.value, "severity": c.severity} for c in active_conditions],
            "condition_count": len(active_conditions),
            "abnormal_lab_count": len(abnormal_labs),
            "abnormal_labs": [{"metric": l.metric_name, "value": l.value, "unit": l.unit} for l in abnormal_labs[:5]],
        },
        "environment": env_data,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


def _fetch_environment_data(recipient) -> dict:
    """Fetch current environmental data for the recipient's location."""
    try:
        from weather import WeatherPredictionModel
        api_key = os.environ.get("WEATHER_API_KEY", "628d4985109c4f6baa3182527250312")
        city = getattr(recipient, 'city', None) or "Delhi"
        model = WeatherPredictionModel(api_key, city)
        data = model.fetch_data(city=city)
        if data and 'current' in data:
            current = data['current']
            aqi = current.get('air_quality', {})
            return {
                "room_temp_c": current.get('temp_c'),
                "humidity": current.get('humidity'),
                "aqi_epa": aqi.get('us-epa-index'),
                "pm25": aqi.get('pm2_5'),
                "location": data.get('location', {}).get('name', city),
            }
    except Exception as e:
        print(f"[insights_engine] Environment fetch failed: {e}")
    return {"room_temp_c": None, "humidity": None, "aqi_epa": None, "pm25": None, "location": None}


# ═══════════════════════════════════════════════════════════════════════
#  RULE-BASED SENSOR FUSION — Cross-domain conclusions
# ═══════════════════════════════════════════════════════════════════════

def generate_fusion_insights(data: dict) -> list:
    """
    Generate cross-domain health insights by combining multiple sensor readings.
    Each insight has: severity, category, title, detail, sensors_involved.
    """
    insights = []
    vitals = data.get("vitals", {})
    audio = data.get("audio", {})
    video = data.get("video", {})
    medical = data.get("medical", {})
    env = data.get("environment", {})
    recipient = data.get("recipient", {})

    hr = vitals.get("heart_rate")
    sbp = vitals.get("systolic_bp")
    dbp = vitals.get("diastolic_bp")
    spo2 = vitals.get("oxygen_saturation")
    temp = vitals.get("temperature")
    sleep = vitals.get("sleep_score")
    bmi = vitals.get("bmi")
    cough_avg = audio.get("cough_daily_avg", 0)
    cough_7d = audio.get("cough_count_7d", 0)
    sneeze_7d = audio.get("sneeze_count_7d", 0)
    fall_count = video.get("fall_count_7d", 0)
    has_fall = video.get("has_recent_fall", False)
    activity = video.get("avg_activity_score", 0)
    mobility = video.get("avg_mobility_score", 0)
    room_temp = env.get("room_temp_c")
    humidity = env.get("humidity")
    aqi = env.get("aqi_epa")
    has_respiratory = recipient.get("respiratory_condition", False)
    conditions = [c["name"].lower() for c in medical.get("active_conditions", [])]
    age = recipient.get("age") or 0

    # ── CROSS-DOMAIN FUSION RULES ──────────────────────────────────

    # 1. HIGH BP + LOW ACTIVITY → Dizziness / Fall Risk
    if sbp and sbp > 140 and activity < 4:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Fall Risk",
            "title": "High Blood Pressure + Low Activity",
            "detail": f"Systolic BP is {sbp} mmHg (elevated) and activity score is only {activity}/10. This combination increases dizziness and fall risk significantly.",
            "recommendation": "Monitor closely. Avoid sudden position changes. Consider BP medication review.",
            "sensors": ["Blood Pressure", "Activity Camera"],
        })

    # 2. LOW SpO2 + HIGH AQI → Respiratory Emergency
    if spo2 and spo2 < 94 and aqi and aqi >= 4:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Respiratory Risk",
            "title": "Low Oxygen + Poor Air Quality",
            "detail": f"SpO2 is {spo2}% (dangerously low) while AQI index is {aqi} (unhealthy). Combined effect poses severe breathing risk.",
            "recommendation": "Move patient to clean air. Use air purifier. Consider supplemental oxygen. Alert caretaker immediately.",
            "sensors": ["SpO2 Sensor", "Air Quality Sensor"],
        })

    # 3. SpO2 LOW + RESPIRATORY CONDITION → Breathing Risk
    if spo2 and spo2 < 95 and has_respiratory:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Respiratory Risk",
            "title": "Low Oxygen with Respiratory Condition",
            "detail": f"SpO2 at {spo2}% in a patient with known respiratory condition. This is below the safe threshold.",
            "recommendation": "Check nebulizer availability. Monitor breathing rate. Consider emergency intervention if SpO2 drops below 90%.",
            "sensors": ["SpO2 Sensor", "Medical History"],
        })

    # 4. HIGH COUGH FREQUENCY + LOW SpO2 → Infection/Pneumonia Risk
    if cough_avg >= THRESHOLDS["cough_daily"]["elevated"] and spo2 and spo2 < 96:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Infection Risk",
            "title": "Persistent Cough + Declining Oxygen",
            "detail": f"Daily cough average is {cough_avg} over 7 days with SpO2 at {spo2}%. Pattern suggests possible respiratory infection or pneumonia.",
            "recommendation": "Schedule chest X-ray. Monitor temperature for fever. Consider antibiotic consultation.",
            "sensors": ["Audio Monitor (Cough)", "SpO2 Sensor"],
        })

    # 5. FEVER + ELEVATED COUGH → Infection Alert
    if temp and temp > 99.5 and cough_7d > 10:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Infection Risk",
            "title": "Fever with Persistent Cough",
            "detail": f"Temperature is {temp}F (elevated) with {cough_7d} cough events in 7 days. Possible upper respiratory infection.",
            "recommendation": "Check for COVID/flu. Monitor temperature trend. Ensure hydration. Consult physician if fever persists > 48h.",
            "sensors": ["Temperature Sensor", "Audio Monitor (Cough)"],
        })

    # 6. FALL DETECTED + HIGH HR → Emergency
    if has_fall and hr and hr > 110:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Emergency",
            "title": "Fall Detected + Elevated Heart Rate",
            "detail": f"Fall detected with heart rate at {hr} bpm (tachycardia). Patient may be in distress or experiencing cardiac event post-fall.",
            "recommendation": "IMMEDIATE CHECK REQUIRED. Verify patient consciousness. Check for injuries. Call emergency services if unresponsive.",
            "sensors": ["Fall Detection Camera", "Heart Rate Sensor"],
        })

    # 7. LOW ACTIVITY + POOR SLEEP → Weakness/Depression Risk
    if sleep and sleep < 50 and activity < 3:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Wellness Risk",
            "title": "Poor Sleep + Very Low Activity",
            "detail": f"Sleep score is {sleep}/100 (poor) with activity at {activity}/10 (very low). Pattern suggests fatigue, weakness, or possible depression.",
            "recommendation": "Encourage gentle movement. Assess sleep environment. Screen for depression. Review sleep medications.",
            "sensors": ["Sleep Sensor", "Activity Camera"],
        })

    # 8. ROOM TOO HOT/COLD + ELDERLY → Heat/Cold Stress
    if room_temp and age >= 60:
        if room_temp > THRESHOLDS["room_temp_c"]["critical_high"]:
            insights.append({
                "severity": SEVERITY_HIGH,
                "category": "Environmental Risk",
                "title": "Room Temperature Dangerously High",
                "detail": f"Room temperature is {room_temp}C ({round(room_temp * 9/5 + 32, 1)}F). Elderly patients are highly susceptible to heat stress and dehydration.",
                "recommendation": "Turn on AC/fan immediately. Ensure patient is hydrated. Move to cooler area if possible.",
                "sensors": ["Room Temperature Sensor"],
            })
        elif room_temp < THRESHOLDS["room_temp_c"]["critical_low"]:
            insights.append({
                "severity": SEVERITY_HIGH,
                "category": "Environmental Risk",
                "title": "Room Temperature Dangerously Low",
                "detail": f"Room temperature is {room_temp}C ({round(room_temp * 9/5 + 32, 1)}F). Risk of hypothermia in elderly patients.",
                "recommendation": "Turn on heating. Add blankets. Monitor patient body temperature closely.",
                "sensors": ["Room Temperature Sensor"],
            })

    # 9. HIGH HUMIDITY + RESPIRATORY → Breathing Difficulty
    if humidity and humidity > 70 and has_respiratory:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Environmental Risk",
            "title": "High Humidity + Respiratory Condition",
            "detail": f"Humidity is {humidity}% (high) for a patient with respiratory issues. High humidity worsens breathing difficulty.",
            "recommendation": "Use dehumidifier. Ensure ventilation. Monitor breathing patterns and SpO2.",
            "sensors": ["Humidity Sensor", "Medical History"],
        })

    # 10. ABNORMAL LABS + FALL HISTORY → Complicated Fall Risk
    if medical.get("abnormal_lab_count", 0) > 2 and fall_count > 0:
        insights.append({
            "severity": SEVERITY_HIGH,
            "category": "Compound Risk",
            "title": "Abnormal Labs + Recent Falls",
            "detail": f"{medical['abnormal_lab_count']} abnormal lab values combined with {fall_count} fall(s) in 7 days. Multiple systems showing stress.",
            "recommendation": "Comprehensive medical review needed. Consider inpatient evaluation. Increase monitoring frequency.",
            "sensors": ["Medical Reports", "Fall Detection Camera"],
        })

    # 11. LOW MOBILITY + HIGH BMI → Pressure Sore Risk
    if bmi and bmi > 30 and mobility < 3:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Mobility Risk",
            "title": "High BMI + Low Mobility",
            "detail": f"BMI is {bmi} (obese) with mobility score of {mobility}/10. High risk of pressure sores, DVT, and muscle atrophy.",
            "recommendation": "Reposition patient every 2 hours. Gentle range-of-motion exercises. Consider physiotherapy.",
            "sensors": ["BMI Data", "Activity Camera"],
        })

    # 12. MULTIPLE CONDITIONS + ELDERLY → Polypharmacy Risk
    if len(conditions) >= 3 and age >= 65:
        insights.append({
            "severity": SEVERITY_MODERATE,
            "category": "Medication Risk",
            "title": "Multiple Conditions in Elderly Patient",
            "detail": f"{len(conditions)} active medical conditions in a {age}-year-old patient. High risk of medication interactions (polypharmacy).",
            "recommendation": "Review all medications for interactions. Consider geriatric medication optimization. Monitor for adverse drug reactions.",
            "sensors": ["Medical Reports", "Age Data"],
        })

    # 13. NO VITALS DATA → Monitoring Gap
    if not vitals.get("recorded_at"):
        insights.append({
            "severity": SEVERITY_LOW,
            "category": "Monitoring Gap",
            "title": "No Recent Vital Signs Data",
            "detail": "No vital signs have been recorded. Sensor connection may be lost or ESP32 device is offline.",
            "recommendation": "Check ESP32 device connection. Verify sensor wiring. Ensure WiFi connectivity.",
            "sensors": ["ESP32 Controller"],
        })

    # 14. TACHYCARDIA + LOW SpO2 → Cardiac Stress
    if hr and hr > 100 and spo2 and spo2 < 94:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Cardiac Risk",
            "title": "Tachycardia with Low Oxygen",
            "detail": f"Heart rate elevated at {hr} bpm with SpO2 at {spo2}%. Heart is compensating for low oxygen — possible cardiac or pulmonary event.",
            "recommendation": "URGENT: Monitor continuously. Prepare for emergency intervention. Alert emergency contacts.",
            "sensors": ["Heart Rate Sensor", "SpO2 Sensor"],
        })

    # 15. HIGH BP + DIABETES → Stroke Risk
    has_diabetes = any("diabetes" in c for c in conditions) or any("sugar" in c or "glucose" in c for c in conditions)
    if sbp and sbp > 150 and has_diabetes:
        insights.append({
            "severity": SEVERITY_CRITICAL,
            "category": "Stroke Risk",
            "title": "Hypertension + Diabetes Combination",
            "detail": f"Systolic BP is {sbp} mmHg with active diabetes. This combination dramatically increases stroke and cardiovascular event risk.",
            "recommendation": "Strict BP control needed. Review diabetic medication. Monitor for signs of stroke (slurred speech, weakness, confusion).",
            "sensors": ["Blood Pressure", "Medical Reports (Diabetes)"],
        })

    # Sort by severity
    severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MODERATE: 2, SEVERITY_LOW: 3, SEVERITY_INFO: 4}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 5))

    return insights


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-SENSOR INSIGHTS — Individual readings outside normal range
# ═══════════════════════════════════════════════════════════════════════

def generate_individual_insights(data: dict) -> list:
    """Generate insights from individual sensor readings that are outside normal ranges."""
    insights = []
    vitals = data.get("vitals", {})
    env = data.get("environment", {})
    audio = data.get("audio", {})

    # Heart Rate
    hr = vitals.get("heart_rate")
    if hr:
        if hr > THRESHOLDS["heart_rate"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critical Heart Rate", "detail": f"Heart rate is {hr} bpm — significantly elevated.", "sensors": ["Heart Rate"]})
        elif hr < THRESHOLDS["heart_rate"]["critical_low"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critically Low Heart Rate", "detail": f"Heart rate is {hr} bpm — bradycardia detected.", "sensors": ["Heart Rate"]})
        elif hr > THRESHOLDS["heart_rate"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Elevated Heart Rate", "detail": f"Heart rate is {hr} bpm — above normal range.", "sensors": ["Heart Rate"]})

    # Blood Pressure
    sbp = vitals.get("systolic_bp")
    if sbp:
        if sbp > THRESHOLDS["systolic_bp"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Hypertensive Crisis", "detail": f"Systolic BP is {sbp} mmHg — emergency level.", "sensors": ["Blood Pressure"]})
        elif sbp > THRESHOLDS["systolic_bp"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "High Blood Pressure", "detail": f"Systolic BP is {sbp} mmHg — elevated.", "sensors": ["Blood Pressure"]})
        elif sbp < THRESHOLDS["systolic_bp"]["critical_low"]:
            insights.append({"severity": SEVERITY_HIGH, "category": "Vitals", "title": "Hypotension", "detail": f"Systolic BP is {sbp} mmHg — dangerously low.", "sensors": ["Blood Pressure"]})

    # SpO2
    spo2 = vitals.get("oxygen_saturation")
    if spo2:
        if spo2 < THRESHOLDS["oxygen_saturation"]["critical_low"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "Critical Oxygen Level", "detail": f"SpO2 is {spo2}% — requires immediate attention.", "sensors": ["SpO2"]})
        elif spo2 < THRESHOLDS["oxygen_saturation"]["low"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Low Oxygen Saturation", "detail": f"SpO2 is {spo2}% — below optimal threshold.", "sensors": ["SpO2"]})

    # Temperature
    temp = vitals.get("temperature")
    if temp:
        if temp > THRESHOLDS["temperature_f"]["critical_high"]:
            insights.append({"severity": SEVERITY_CRITICAL, "category": "Vitals", "title": "High Fever", "detail": f"Temperature is {temp}F — high fever. Investigate infection.", "sensors": ["Temperature"]})
        elif temp > THRESHOLDS["temperature_f"]["high"]:
            insights.append({"severity": SEVERITY_MODERATE, "category": "Vitals", "title": "Mild Fever", "detail": f"Temperature is {temp}F — above normal.", "sensors": ["Temperature"]})

    # Cough Frequency
    cough_avg = audio.get("cough_daily_avg", 0)
    if cough_avg >= THRESHOLDS["cough_daily"]["critical"]:
        insights.append({"severity": SEVERITY_HIGH, "category": "Respiratory", "title": "Very High Cough Frequency", "detail": f"Average {cough_avg} coughs/day over 7 days — investigate respiratory infection.", "sensors": ["Audio Monitor"]})
    elif cough_avg >= THRESHOLDS["cough_daily"]["elevated"]:
        insights.append({"severity": SEVERITY_LOW, "category": "Respiratory", "title": "Elevated Cough Frequency", "detail": f"Average {cough_avg} coughs/day — monitor for worsening.", "sensors": ["Audio Monitor"]})

    return insights


# ═══════════════════════════════════════════════════════════════════════
#  GEMINI AI — Deep clinical interpretation of fused data
# ═══════════════════════════════════════════════════════════════════════

def gemini_fused_analysis(data: dict, fusion_insights: list) -> dict:
    """
    Send ALL aggregated patient data + rule-based insights to Gemini
    for deep clinical interpretation, actionable summary, and care plan.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[insights_engine] GEMINI_API_KEY not set - returning rule-based insights only")
        return _fallback_summary(data, fusion_insights)

    # Build a comprehensive prompt
    prompt = f"""You are a senior clinical AI analyst for an elderly care monitoring system called Caretaker.ai.
You have access to REAL-TIME data from multiple sensors monitoring an elderly patient.

PATIENT DATA (all sensors, real-time):
{json.dumps(data, indent=2, default=str)}

RULE-BASED INSIGHTS (already generated by our deterministic engine):
{json.dumps(fusion_insights, indent=2, default=str)}

YOUR TASK — Generate a comprehensive clinical analysis. Return a JSON object with EXACTLY these keys:

{{
  "overall_status": "1-2 sentence overall patient status (use medical terminology)",
  "risk_level": "low/moderate/high/critical",
  "key_concerns": [
    {{"concern": "Brief title", "explanation": "Why this matters", "action": "What to do"}}
  ],
  "sensor_correlations": [
    {{"combination": "Sensor A + Sensor B", "finding": "What the combination tells us", "clinical_significance": "Why it matters"}}
  ],
  "immediate_actions": ["Action 1 to take now", "Action 2"],
  "care_plan": {{
    "daily": ["Daily monitoring tasks"],
    "weekly": ["Weekly checkups"],
    "medical": ["Medical appointments to schedule"]
  }},
  "vitals_interpretation": {{
    "heart_rate": "Normal/Elevated/Low — clinical meaning",
    "blood_pressure": "Status and risk",
    "oxygen": "Status and risk",
    "temperature": "Status and risk"
  }},
  "environment_assessment": "Are living conditions safe? What to improve?",
  "activity_assessment": "Is the patient active enough? Risks from inactivity?",
  "prognosis": "Short-term outlook (next 1-2 weeks) based on all data"
}}

Rules:
- Be specific with numbers from the data
- If a sensor has no data (null), say "No data available — recommend connecting [sensor]"
- Consider the patient's age ({data.get('recipient', {}).get('age', 'unknown')}) and existing conditions
- Focus on actionable insights, not generic advice
- Use clinical language appropriate for a healthcare professional
"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        resp = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 3000, "temperature": 0.3}
        }, headers={"Content-Type": "application/json"}, timeout=90)

        if resp.status_code == 200:
            result_data = resp.json()
            if "candidates" in result_data and result_data["candidates"]:
                raw = result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"[insights_engine] Gemini response length: {len(raw)} chars")
                import re
                cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.DOTALL)
                cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except json.JSONDecodeError:
                            pass
                    print(f"[insights_engine] Failed to parse Gemini response")

        print(f"[insights_engine] Gemini returned status {resp.status_code}")
    except Exception as e:
        print(f"[insights_engine] Gemini call failed: {e}")

    return _fallback_summary(data, fusion_insights)


def _fallback_summary(data: dict, insights: list) -> dict:
    """Deterministic fallback when Gemini is unavailable."""
    critical = [i for i in insights if i["severity"] == SEVERITY_CRITICAL]
    high = [i for i in insights if i["severity"] == SEVERITY_HIGH]

    if critical:
        level = "critical"
        status = f"CRITICAL: {len(critical)} critical issue(s) detected requiring immediate attention."
    elif high:
        level = "high"
        status = f"HIGH RISK: {len(high)} significant concern(s) identified."
    elif insights:
        level = "moderate"
        status = "Moderate concerns detected. Continue monitoring."
    else:
        level = "low"
        status = "All monitored parameters within acceptable ranges."

    return {
        "overall_status": status,
        "risk_level": level,
        "key_concerns": [{"concern": i["title"], "explanation": i["detail"], "action": i.get("recommendation", "Monitor closely")} for i in insights[:5]],
        "immediate_actions": [i.get("recommendation", i["detail"]) for i in critical[:3]] if critical else ["Continue routine monitoring"],
        "prognosis": "Unable to generate detailed prognosis (AI service unavailable). Rule-based analysis active.",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — Called by the API route
# ═══════════════════════════════════════════════════════════════════════

def generate_full_insights(recipient_id: int, db: Session) -> dict:
    """
    Main entry point: Aggregate → Fuse → Analyze → Return.
    Returns the complete patient intelligence report.
    """
    print(f"[insights_engine] Generating full insights for recipient {recipient_id}")

    # Step 1: Aggregate all data sources
    data = aggregate_patient_data(recipient_id, db)
    if "error" in data:
        return data

    # Step 2: Rule-based sensor fusion (fast, deterministic)
    fusion_insights = generate_fusion_insights(data)
    individual_insights = generate_individual_insights(data)

    # Step 3: Gemini AI deep interpretation
    ai_analysis = gemini_fused_analysis(data, fusion_insights)

    # Step 4: Combine everything
    result = {
        "patient_data": data,
        "fusion_insights": fusion_insights,
        "individual_insights": individual_insights,
        "ai_analysis": ai_analysis,
        "total_insights": len(fusion_insights) + len(individual_insights),
        "critical_count": len([i for i in fusion_insights + individual_insights if i["severity"] == SEVERITY_CRITICAL]),
        "high_count": len([i for i in fusion_insights + individual_insights if i["severity"] == SEVERITY_HIGH]),
        "generated_at": datetime.datetime.utcnow().isoformat(),
    }

    print(f"[insights_engine] Generated {result['total_insights']} insights ({result['critical_count']} critical, {result['high_count']} high)")
    return result
