"""Report Ingestion Service — Structured data extraction from medical reports via Gemini.

Extracts: report_date, diagnoses, lab_values, medications, doctor_notes, symptoms
from raw medical report text. Falls back gracefully if Gemini is unavailable.
"""

import os
import json
import re
from datetime import date
from dotenv import load_dotenv

load_dotenv()

try:
    import requests
except ImportError:
    requests = None


def extract_structured_report(text: str, report_date_hint: str = None) -> dict:
    """Extract structured medical data from report text using Gemini.

    Args:
        text: Raw text extracted from the medical report
        report_date_hint: Optional date hint from filename or upload date (YYYY-MM-DD)

    Returns:
        dict with keys: report_date, diagnoses, lab_values, medications, doctor_notes, symptoms
    """
    empty_result = {
        "report_date": report_date_hint or str(date.today()),
        "diagnoses": [],
        "resolved_diagnoses": [],
        "lab_values": {},
        "medications": [],
        "doctor_notes": "",
        "symptoms": []
    }

    if not text or not text.strip():
        print("[report_ingestion] No text to extract from")
        return empty_result

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not requests:
        print("[report_ingestion] Gemini not configured; returning empty structure")
        return empty_result

    prompt = """You are a clinical data extractor. Extract STRUCTURED medical information from the following medical report.

Return a VALID JSON object with EXACTLY this structure (no markdown, no code fences, just raw JSON):
{
  "report_date": "YYYY-MM-DD",
  "diagnoses": ["Disease Name 1", "Disease Name 2"],
  "resolved_diagnoses": ["Cured Disease 1"],
  "lab_values": {
    "MetricName": {"value": 8.2, "unit": "%"},
    "Systolic BP": {"value": 160, "unit": "mmHg"},
    "Diastolic BP": {"value": 95, "unit": "mmHg"}
  },
  "medications": ["Medication 1 with dose", "Medication 2"],
  "doctor_notes": "Key observations from the doctor",
  "symptoms": ["Symptom 1", "Symptom 2"]
}

RULES:
1. Use standard metric names: HbA1c, Fasting Glucose, Systolic BP, Diastolic BP, Total Cholesterol, LDL, HDL, Triglycerides, Creatinine, eGFR, Hemoglobin, TSH, T3, T4, Uric Acid, BUN, Hematocrit, Ferritin
2. For blood pressure written as "160/95", split into "Systolic BP" and "Diastolic BP"
3. For report_date, extract the actual report/visit date from the document. If not found, use """ + f'"{report_date_hint or str(date.today())}"' + """
4. Include ALL lab values found, even if they appear normal
5. Only include diagnoses that are explicitly mentioned or clearly implied
6. Return ONLY valid JSON, no other text

Medical Report:
""" + text[:8000]  # Limit text to avoid token overflow

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 2000,
                "temperature": 0.1,
                "topP": 0.8
            }
        }

        print(f"[report_ingestion] Calling Gemini for structured extraction ({len(text)} chars)")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data and data["candidates"]:
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

                # Clean markdown code fences if present (multiline-safe)
                raw_text = raw_text.strip()
                cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.DOTALL)
                cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()

                try:
                    result = json.loads(cleaned)
                    # Validate structure
                    result.setdefault("report_date", report_date_hint or str(date.today()))
                    result.setdefault("diagnoses", [])
                    result.setdefault("resolved_diagnoses", [])
                    result.setdefault("lab_values", {})
                    result.setdefault("medications", [])
                    result.setdefault("doctor_notes", "")
                    result.setdefault("symptoms", [])
                    print(f"[report_ingestion] Extracted: {len(result['diagnoses'])} diagnoses, {len(result['lab_values'])} lab values")
                    return result
                except json.JSONDecodeError:
                    # Fallback: try extracting first { ... } block
                    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                    if match:
                        try:
                            result = json.loads(match.group(0))
                            result.setdefault("report_date", report_date_hint or str(date.today()))
                            result.setdefault("diagnoses", [])
                            result.setdefault("resolved_diagnoses", [])
                            result.setdefault("lab_values", {})
                            result.setdefault("medications", [])
                            result.setdefault("doctor_notes", "")
                            result.setdefault("symptoms", [])
                            print(f"[report_ingestion] Extracted (fallback): {len(result['diagnoses'])} diagnoses, {len(result['lab_values'])} lab values")
                            return result
                        except json.JSONDecodeError:
                            pass
                    print(f"[report_ingestion] Failed to parse Gemini JSON response")
                    print(f"[report_ingestion] Raw response: {raw_text[:500]}")
                    return empty_result
        else:
            print(f"[report_ingestion] Gemini returned status {resp.status_code}")
            return empty_result

    except Exception as e:
        print(f"[report_ingestion] Extraction failed: {e}")
        return empty_result
