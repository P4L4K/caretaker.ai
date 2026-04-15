"""Report Ingestion Service v2 — Hybrid Pipeline.

Pipeline order:
  1. Text extraction          (pdfplumber / OCR — callers responsibility)
  2. Date extraction          (deterministic regex → date_extractor)
  3. Document cleaning        (dedup, skip-lines, row reconstruction → document_cleaner)
  4. Lab parsing              (template detect → section split → rule/fuzzy extraction → LLM fallback → lab_parser)
  5. Normalization/Validation (canonical names, unit conversion, physiological guards → lab_normalizer)
  6. LabValue persistence     (save to DB with source_text + confidence_score)
  7. Legacy Gemini extraction (for diagnoses, medications, doctor_notes — still useful)
  8. RAG indexing             (async — non-blocking → rag_service)

The old `extract_structured_report` function is kept for backwards compatibility but
now only handles diagnoses/medications/notes (NOT lab values — those come from the
new hybrid pipeline).
"""
from __future__ import annotations

import os
import json
import re
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from utils.gemini_client import call_gemini

load_dotenv(override=True)

try:
    import requests
except ImportError:
    requests = None


# ─── New Pipeline ─────────────────────────────────────────────────────────────

def run_hybrid_lab_extraction(
    text: str,
    report_id: int,
    care_recipient_id: int,
    upload_date: Optional[date] = None,
    db=None,
) -> dict:
    """Run the full hybrid lab extraction pipeline on report text.

    Args:
        text:               Raw extracted text from the report.
        report_id:          DB ID of the MedicalReport record.
        care_recipient_id:  Recipient's DB ID.
        upload_date:        Upload date to use as date fallback.
        db:                 SQLAlchemy session (for saving LabValues).

    Returns:
        {
          "report_date":    date | None,
          "template":       str,
          "lab_rows":       list[dict],   # Validated, normalized rows ready for DB
          "unresolved":     list[str],    # Lines that couldn't be parsed
          "sections":       list[str],
          "saved_count":    int,          # How many LabValues were saved
        }
    """
    from services.date_extractor import extract_date_from_text
    from services.document_cleaner import clean_document
    from services.lab_parser import parse_report
    from services.lab_normalizer import normalize_and_validate

    # ── Step 1: Deterministic date extraction ─────────────────────────────────
    report_date = extract_date_from_text(text, fallback=upload_date or date.today())
    print(f"[ingestion_v2] Report date: {report_date}")

    # ── Step 2: Parse (clean + section + extract + LLM fallback) ─────────────
    parsed = parse_report(text)
    template   = parsed["template"]
    raw_rows   = parsed["rows"]
    unresolved = parsed["unresolved"]
    sections   = parsed["section_labels"]
    cleaned    = clean_document(text)

    print(f"[ingestion_v2] Template={template}, raw_rows={len(raw_rows)}")

    # ── Step 3: Normalize + validate each row ─────────────────────────────────
    validated_rows: list[dict] = []
    for row in raw_rows:
        norm = normalize_and_validate(
            raw_name=row.get("raw_name", ""),
            raw_value=row.get("value", 0.0),
            raw_unit=row.get("unit", ""),
            source=row.get("source", "regex"),
        )
        if norm is None:
            continue   # Rejected by physiological guard or unmappable name
        norm.update({
            "source_text":   row.get("source_text", ""),
            "ref_low":       row.get("ref_low"),
            "ref_high":      row.get("ref_high"),
            "section":       row.get("section", "GENERAL"),
            "report_date":   report_date,
        })
        validated_rows.append(norm)

    print(f"[ingestion_v2] Validated: {len(validated_rows)} / {len(raw_rows)} rows")

    # ── Step 4: Save LabValues to DB ──────────────────────────────────────────
    saved_count = 0
    if db is not None and validated_rows:
        saved_count = _save_lab_values(
            db, validated_rows, report_id, care_recipient_id, report_date
        )

    # ── Step 5: RAG indexing (best-effort, non-blocking) ─────────────────────
    try:
        from services.rag_service import index_report_chunks
        index_report_chunks(
            report_id=report_id,
            care_recipient_id=care_recipient_id,
            cleaned_lines=cleaned["cleaned_lines"],
            extracted_rows=raw_rows,
        )
    except Exception as e:
        print(f"[ingestion_v2] RAG indexing failed (non-fatal): {e}")

    return {
        "report_date":  report_date,
        "template":     template,
        "lab_rows":     validated_rows,
        "unresolved":   unresolved,
        "sections":     sections,
        "saved_count":  saved_count,
    }


def _save_lab_values(
    db,
    validated_rows: list[dict],
    report_id: int,
    care_recipient_id: int,
    report_date: date,
) -> int:
    """Persist validated lab rows to the LabValue table.
    
    Deduplicates: if a row with the same (metric_name, recorded_date, report_id)
    already exists, it is skipped.
    
    Returns number of rows saved.
    """
    from tables.medical_conditions import LabValue

    saved = 0
    for row in validated_rows:
        metric = row["metric_name"]
        rec_date = row.get("report_date") or report_date

        # Duplicate check
        existing = db.query(LabValue).filter(
            LabValue.care_recipient_id == care_recipient_id,
            LabValue.metric_name == metric,
            LabValue.recorded_date == rec_date,
            LabValue.report_id == report_id,
        ).first()
        if existing:
            continue

        # Abnormal detection
        ref_low  = row.get("ref_low")
        ref_high = row.get("ref_high")
        norm_val = row["normalized_value"]
        is_abnormal = False
        if ref_low is not None and norm_val < ref_low:
            is_abnormal = True
        if ref_high is not None and norm_val > ref_high:
            is_abnormal = True

        lv = LabValue(
            care_recipient_id   = care_recipient_id,
            report_id           = report_id,
            metric_name         = metric,
            metric_value        = row["metric_value"],
            unit                = row["unit"],
            normalized_value    = norm_val,
            normalized_unit     = row["normalized_unit"],
            reference_range_low = ref_low,
            reference_range_high= ref_high,
            is_abnormal         = is_abnormal,
            recorded_date       = rec_date,
            source_text         = row.get("source_text", "")[:500] if row.get("source_text") else None,
            confidence_score    = row.get("confidence_score", 0.9),
            extraction_source   = row.get("extraction_source", "regex"),
        )
        db.add(lv)
        saved += 1

    if saved:
        db.flush()

    return saved


# ─── Legacy Gemini Extraction (Diagnoses, Medications, Notes) ─────────────────
# This is now ONLY responsible for non-numeric clinical context extraction.
# Lab values are fully handled by the hybrid pipeline above.

def extract_structured_report(text: str, report_date_hint: str = None) -> dict:
    """Extract diagnoses, medications, doctor_notes, and symptoms via Gemini.

    Lab values are no longer extracted here — they come from run_hybrid_lab_extraction().
    This function now fills the 'clinical context' part of the extracted_data JSON.
    """
    empty_result = {
        "report_date": report_date_hint or str(date.today()),
        "diagnoses": [],
        "resolved_diagnoses": [],
        "lab_values": {},       # Left empty — hybrid pipeline handles this
        "medications": [],
        "doctor_notes": "",
        "symptoms": [],
    }

    if not text or not text.strip():
        return empty_result

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not requests:
        return empty_result

    # Truncate text to essentials (reduce token cost)
    # For diagnosis detection we mostly need the first and last sections
    truncated = text[:4000] + ("\n...\n" + text[-2000:] if len(text) > 6000 else "")

    prompt = """You are a clinical data extractor. From the medical report below, extract ONLY:
- Active diagnoses / diseases explicitly mentioned
- Resolved / past diagnoses mentioned
- Medications listed (name + dose if available)
- Key doctor observations or impressions (brief)
- Symptoms reported by the patient

Return a VALID JSON object with EXACTLY this structure (no markdown, no code fences):
{
  "report_date": "YYYY-MM-DD or empty string",
  "diagnoses": ["Disease Name 1"],
  "resolved_diagnoses": ["Past Disease"],
  "lab_values": {},
  "medications": ["Medication 1 with dose"],
  "doctor_notes": "Brief clinical impressions",
  "symptoms": ["Symptom 1"]
}

RULES:
1. Leave lab_values as empty {} — they are handled separately
2. Only include diagnoses explicitly mentioned or clearly implied
3. Return null fields as empty lists/strings, NOT as null
4. Return ONLY valid JSON, no other text

Medical Report:
""" + truncated

    try:
        print(f"[ingestion_v2] Gemini clinical context extraction ({len(text)} chars)")
        data = call_gemini(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.05, "topP": 0.8},
            },
            timeout=60,
            caller="[ingestion_v2/gemini_context]",
        )

        if data and "candidates" in data and data["candidates"]:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Clean markdown code fences if present (multiline-safe)
            raw_text = raw_text.strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.DOTALL)
            cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()

            try:
                result = json.loads(cleaned)
                # Ensure date is always valid
                if not result.get("report_date") or not str(result["report_date"]).strip():
                    result["report_date"] = report_date_hint or str(date.today())
                
                # Validate structure
                result.setdefault("diagnoses", [])
                result.setdefault("resolved_diagnoses", [])
                result["lab_values"] = {}   # Always override — we don't want Gemini lab guesses
                result.setdefault("medications", [])
                result.setdefault("doctor_notes", "")
                result.setdefault("symptoms", [])
                print(f"[ingestion_v2] Gemini context: {len(result['diagnoses'])} diagnoses")
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
                        result["lab_values"] = {}
                        result.setdefault("medications", [])
                        result.setdefault("doctor_notes", "")
                        result.setdefault("symptoms", [])
                        return result
                    except json.JSONDecodeError:
                        pass
                print(f"[ingestion_v2] Failed to parse Gemini JSON response")
        return empty_result

    except Exception as e:
        print(f"[ingestion_v2] Gemini context extraction failed: {e}")
        return empty_result


def extract_text_from_image_bytes(image_bytes: bytes, mime_type: str, report_date_hint: str = None) -> dict:
    """Extract structured medical data directly from an image using Gemini Vision."""
    import base64
    empty_result = {
        "report_date": report_date_hint or str(date.today()),
        "diagnoses": [],
        "resolved_diagnoses": [],
        "lab_values": {},
        "medications": [],
        "doctor_notes": "",
        "symptoms": []
    }
    if not image_bytes: return empty_result
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return empty_result

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = """Extract medical info from image. Return JSON: {report_date, diagnoses, resolved_diagnoses, lab_values, medications, doctor_notes, symptoms}."""

    try:
        payload = {
            "contents": [{
                "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": b64_image}}]
            }],
            "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.1, "topP": 0.8}
        }
        data = call_gemini(payload, timeout=90, caller="[report_ingestion/ocr]")
        if data and "candidates" in data and data["candidates"]:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.DOTALL)
            cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()
            try:
                result = json.loads(cleaned)
                for k in empty_result: result.setdefault(k, empty_result[k])
                return result
            except json.JSONDecodeError:
                pass
        return empty_result
    except Exception as e:
        print(f"[report_ingestion/ocr] Extraction failed: {e}")
        return empty_result
