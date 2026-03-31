import os
from io import BytesIO
import re
from dotenv import load_dotenv
from utils.gemini_client import call_gemini

# Ensure environmental variables are loaded
load_dotenv(override=True)

try:
    import requests
except Exception:
    requests = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pdf2image
except Exception:
    pdf2image = None

try:
    import docx
except Exception:
    docx = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None


def _simple_text_from_pdf_bytes(b: bytes):
    """Extract text from PDF bytes, with OCR fallback."""
    text = ''
    # 1. Try with pdfplumber - often good for structured PDFs
    if pdfplumber:
        try:
            with pdfplumber.open(BytesIO(b)) as pdf:
                parts = [p.extract_text() or '' for p in pdf.pages]
                text = '\n'.join(parts).strip()
        except Exception:
            text = ''  # ignore failures

    # 2. If no text, try PyPDF2
    if not text and PdfReader:
        try:
            reader = PdfReader(BytesIO(b))
            parts = [page.extract_text() or '' for page in reader.pages]
            text = '\n'.join(parts).strip()
        except Exception:
            text = ''  # ignore failures

    # 3. If still no text, and this is a PDF, try OCR
    if not text and pdf2image:
        print('[summarizer] PDF text extraction failed, trying OCR fallback...')
        try:
            images = pdf2image.convert_from_bytes(b)
            parts = []
            for i, image in enumerate(images):
                try:
                    # TODO: consider adding language hints if known
                    parts.append(pytesseract.image_to_string(image) or '')
                except Exception as e:
                    print(f'[summarizer] OCR on page {i} failed: {e}')
            text = '\n'.join(parts).strip()
            print(f'[summarizer] OCR fallback produced {len(text)} chars')
        except Exception as e:
            # This can happen if poppler is not installed
            print(f'[summarizer] OCR fallback failed entirely: {e}')
            text = ''

    return text


def _simple_text_from_docx_bytes(b: bytes):
    if not docx:
        return ''
    try:
        doc = docx.Document(BytesIO(b))
        paragraphs = [p.text for p in doc.paragraphs]
        return '\n'.join(paragraphs)
    except Exception:
        return ''


def _simple_text_from_image_bytes(b: bytes):
    if not Image or not pytesseract:
        return ''
    try:
        img = Image.open(BytesIO(b))
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ''


def extract_text_from_bytes(b: bytes, mime: str = None) -> str:
    """Attempt to extract text from bytes based on mime type. Returns empty string if unable."""
    mime = (mime or '').lower()
    if 'pdf' in mime:
        return _simple_text_from_pdf_bytes(b)
    if 'word' in mime or 'officedocument' in mime or mime.endswith('.docx'):
        return _simple_text_from_docx_bytes(b)
    if mime.startswith('image'):
        return _simple_text_from_image_bytes(b)

    # Last-ditch: try to decode as utf-8 text
    try:
        text = b.decode('utf-8')
        # if text looks binary, return empty
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            return ''
        return text
    except Exception:
        return ''


def extract_clinical_findings(medical_text: str) -> str:
    """Extract key clinical information from medical documents using Gemini.
    
    Focuses on:
    - Diagnoses / diseases
    - Symptoms and complaints
    - Abnormal test results
    - Important vitals (if high/low)
    - Medications prescribed (with dose if mentioned)
    - Relevant dates
    - Doctor assessments or impressions
    - Follow-up or recommended care
    
    Returns extracted findings without filler language.
    """
    if not medical_text:
        return ''

    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key or not requests:
        print('[summarizer] Gemini API key not configured; cannot extract clinical findings')
        return ''

    clinical_extraction_prompt = """You are a clinical information extractor. Summarize the following medical documents into clear, precise medical findings.

Focus ONLY on:
• Diagnoses / diseases the patient has
• Symptoms and complaints
• Abnormal test results
• Important vitals (if high/low)
• Medications prescribed (with dose if mentioned)
• Relevant dates
• Doctor assessments or impressions
• Any follow-up or recommended care

Do NOT include filler language.
Do NOT rewrite entire paragraphs.
Do NOT add content not found in the report.
Just extract key medical facts and present them clearly.

Medical Document:
{text}"""

    try:
        prompt = clinical_extraction_prompt.format(text=medical_text)
        
        # Use Google Generative AI library if available, otherwise fall back to REST API
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            print('[summarizer] Clinical extraction via genai library successful')
            return response.text if response.text else ''
        except Exception as e:
            print(f'[summarizer] genai library not available ({e}), trying REST API...')
            
            # Fall back to REST API
            headers = {
                'Content-Type': 'application/json'
            }
            payload = {
                'contents': [
                    {
                        'parts': [
                            {
                                'text': prompt
                            }
                        ]
                    }
                ]
            }
            
            print(f'[summarizer] Calling Gemini REST API for clinical extraction')
            data = call_gemini(payload, timeout=60, caller="[summarizer/clinical]")
            if data:
                try:
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if len(parts) > 0 and 'text' in parts[0]:
                                return parts[0]['text']
                except Exception as e:
                    print(f'[summarizer] Failed to parse Gemini response: {e}')
                    return ''
            return ''
                
    except Exception as e:
        print(f'[summarizer] Clinical extraction failed: {e}')
        return ''


def summarize_text_via_gemini(text: str, target_words: int = 250) -> str:
    """Call out to an external Gemini endpoint (config via env) to get a summary of approximately target_words.
    If not configured, fall back to a simple local summarizer.
    """
    if not text:
        return ''

    endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    api_key = os.environ.get('GEMINI_API_KEY')

    # Basic local fallback summarizer: take sentences until target_words reached
    def local_summary(s: str) -> str:
        if not s:
            return ''
        sentences = re.split(r'(?<=[.!?])\s+', s.strip())
        out_words = []
        for sent in sentences:
            parts = sent.split()
            if not parts:
                continue
            # if adding this sentence goes over target, stop and break
            if len(out_words) + len(parts) > target_words:
                break
            out_words.extend(parts)
            if len(out_words) >= target_words:
                break
        if not out_words:
            # fallback: just take the first N words from the raw text
            return ' '.join(s.split()[:target_words])
        return ' '.join(out_words)

    if not endpoint or not api_key or not requests:
        print('[summarizer] Gemini not configured or requests missing; using local summary fallback')
        return local_summary(text)

    try:
        # Log the input text for debugging
        print(f'[summarizer] Input text length: {len(text)} characters')
        if len(text) > 200:
            print(f'[summarizer] Text preview: {text[:100]}...{text[-100:]}')
        else:
            print(f'[summarizer] Text: {text}')

        # Format the prompt with the actual text - MIND MAP STRUCTURE
        prompt = """You are a clinical information extractor and health-risk analyst.

Extract critical medical information and present it in well-defined sections. 
CRITICAL RULES:
1. For each section, provide 2-4 distinct bullet points. 
2. Use '• ' for category titles (e.g., • Diagnoses: ).
3. Use '  - ' for sub-points under those titles.
4. Do NOT use commas to separate different items; always use a new sub-point line.
5. Do NOT include any text outside these titles and points.

START_CLINICAL_SUMMARY
• Diagnoses: 
  - [Point 1]
  - [Point 2]
• Symptoms & History: 
  - [Point 1]
  - [Point 2]
• Vitals & Labs: 
  - [Point 1]
  - [Point 2]
• Medications: 
  - [Point 1]
  - [Point 2]
• Care Plan: 
  - [Point 1]
  - [Point 2]
END_CLINICAL_SUMMARY

START_ENVIRONMENTAL_THRESHOLD
1. Recommended Temperature Range (°C): [Value]
2. Recommended Relative Humidity Range (%): [Value]
3. Recommended Indoor Air Quality (PM2.5 / AQI): [Value]
END_ENVIRONMENTAL_THRESHOLD

========================
Medical Document Text:
{text}""".format(text=text.strip())
        
        # Structure the payload according to Gemini API spec
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2000,
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        print(f'[summarizer] Calling Gemini (payload words approx {len(prompt.split())})')
        data = call_gemini(payload, timeout=60, caller="[summarizer/text]")
        if data:
            try:
                if 'candidates' in data and data['candidates']:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts and 'text' in parts[0]:
                            full_response = parts[0]['text']
                            
                            # Split response into summary and thresholds
                            if "ENVIRONMENTAL THRESHOLDS" in full_response:
                                summary, thresholds_section = full_response.split("ENVIRONMENTAL THRESHOLDS", 1)
                                thresholds = parse_environmental_thresholds("ENVIRONMENTAL THRESHOLDS" + thresholds_section)
                                
                                # Add thresholds to summary
                                summary += "\n\nENVIRONMENTAL RECOMMENDATIONS:\n"
                                if thresholds["temperature"]:
                                    summary += f"• Temperature: {thresholds['temperature']}\n"
                                if thresholds["humidity"]:
                                    summary += f"• Humidity: {thresholds['humidity']}\n"
                                if thresholds["air_quality"]:
                                    summary += f"• Air Quality: {thresholds['air_quality']}\n"
                                
                                for rec in thresholds["additional_recommendations"]:
                                    summary += f"• {rec}\n"
                            else:
                                summary = full_response
                            
                            # Return the full response without truncation
                            print(f'[summarizer] Gemini produced {len(summary.split())} words')
                            
                            # Ensure proper line breaks for bullet points
                            summary = summary.replace('•', '\n•')  # Add newline before each bullet
                            summary = '\n'.join(line.strip() for line in summary.split('\n'))  # Clean up whitespace
                            
                            return summary
            except Exception as e:
                print(f'[summarizer] Failed to parse Gemini response: {e}')
                return local_summary(text)
        return local_summary(text)
        
    except Exception as e:
        print(f'[summarizer] Error in summarize_text_via_gemini: {str(e)}')
        return local_summary(text)

def parse_environmental_thresholds(thresholds_text: str) -> dict:
    """Parse environmental thresholds from the Gemini response.
    
    Args:
        thresholds_text: Raw text containing the environmental thresholds section
        
    Returns:
        dict: Parsed thresholds with keys: temperature, humidity, air_quality, additional_recommendations
    """
    thresholds = {
        "temperature": None,
        "humidity": None,
        "air_quality": None,
        "additional_recommendations": []
    }
    
    if not thresholds_text:
        return thresholds
    
    # For the new concise format with numbered items
    if '1. Recommended Temperature' in thresholds_text:
        lines = thresholds_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('1.'):
                temp = line.split(':', 1)[-1].strip()
                if '°C' not in temp and 'C' in temp:  # Handle case where ° is missing
                    temp = temp.replace('C', '°C')
                thresholds["temperature"] = temp
            elif line.startswith('2.'):
                thresholds["humidity"] = line.split(':', 1)[-1].strip()
            elif line.startswith('3.'):
                thresholds["air_quality"] = line.split(':', 1)[-1].strip()
    else:
        # Fallback to old format parsing
        # Extract temperature (supports formats like "20-24°C" or "20°C to 24°C")
        temp_match = re.search(r'(\d+)\s*°?C?\s*[\-\s]+\s*(\d+)\s*°?C', thresholds_text)
        if temp_match:
            min_temp, max_temp = temp_match.groups()
            thresholds["temperature"] = f"{min_temp}°C - {max_temp}°C"
        
        # Extract humidity (supports formats like "40-55%" or "40% to 55%")
        hum_match = re.search(r'(\d+)\s*[%]\s*[-\s]+\s*(\d+)\s*%', thresholds_text)
        if hum_match:
            min_hum, max_hum = hum_match.groups()
            thresholds["humidity"] = f"{min_hum}% - {max_hum}%"
        
        # Extract air quality (looks for AQI or PM2.5 values)
        aqi_match = re.search(r'(?:AQI|PM2\.5)\s*[:\-]?\s*([^.\n]+)', thresholds_text, re.IGNORECASE)
        if aqi_match:
            thresholds["air_quality"] = aqi_match.group(1).strip()
    
    return thresholds
def summarize_report_insights_via_gemini(insights_list: list) -> str:
    """Call out to Gemini to get a combined historical analysis BASED on individual report insights.
    Focuses on progression, patterns, and aggregate trends.
    """
    if not insights_list:
        return ''

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key or not requests:
        print('[summarizer] Gemini not configured for aggregate summary.')
        return 'Unable to generate aggregate summary without API key.'

    # Construct the context for Gemini
    history_text = ""
    for idx, insight in enumerate(insights_list):
        history_text += f"--- REPORT {idx+1} INSIGHTS ---\n{insight}\n\n"

    prompt = """You are a Senior Clinical Analyst specialized in longitudinal health history.
You will receive several summaries from different medical reports for a single patient.

Your task is to synthesize these individual snapshots into a PREMIER AGGREGATE HEALTH STORY.
CRITICAL GOALS:
1. Identify clinical progression (e.g., "Symptoms have worsened since Report 1").
2. Identify chronic vs acute conditions (e.g., "Patient consistently shows high BP across all reports").
3. Detect medication conflicts or adjustments.
4. Refine environmental risks based on the TOTAL medical picture.

Use the structure EXACTLY as below:

START_CLINICAL_SUMMARY
• Diagnosis Timeline: 
  - [Point 1: Long-term vs New diagnoses]
  - [Point 2: Progression of conditions]
• Symptom Trends: 
  - [Point 1: Patterns or changes observed]
• Vital Signs History: 
  - [Point 1: Abnormal trends across multiple reports]
• Medication Management: 
  - [Point 1: Active meds and history of changes]
• Integrated Care Plan: 
  - [Point 1: Combined follow-up strategy]
  - [Point 2: Priority actions]
END_CLINICAL_SUMMARY

START_ENVIRONMENTAL_THRESHOLD
1. Recommended Temperature Range (°C): [Value optimized for history]
2. Recommended Relative Humidity Range (%): [Value optimized for history]
3. Recommended Indoor Air Quality (PM2.5 / AQI): [Value optimized for history]
END_ENVIRONMENTAL_THRESHOLD

Rules:
- Be concise (2-4 points per section).
- Focus on CROSS-REPORT analysis.
- Use '• ' for titles and '  - ' for sub-points.

========================
INDIVIDUAL REPORT SUMMARIES:
{text}""".format(text=history_text.strip())

    try:
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.2}
        }
        print(f'[summarizer] Calling Gemini for AGGREGATE summary (history_text_len={len(history_text)})')
        data = call_gemini(payload, timeout=60, caller="[summarizer/aggregate]")
        if data and 'candidates' in data and len(data['candidates']) > 0:
            summary = data['candidates'][0]['content']['parts'][0]['text']
            summary = summary.replace('•', '\n•')
            summary = '\n'.join(line.strip() for line in summary.split('\n'))
            print(f'[summarizer] Aggregate summary produced: {len(summary.split())} words')
            return summary
        return "Failed to synthesize aggregate history."
    except Exception as e:
        print(f'[summarizer] Aggregate synthesis failed: {e}')
        return "Synthesis error."

def chat_with_patient_ai(patient_context: dict, chat_history: list, user_message: str) -> str:
    """Answers a doctor's question about a specific patient using Gemini 2.5 Flash with full clinical context."""
    import json
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key or not requests:
        return 'Unable to use AI Clinical Chat without Gemini API key.'

    # Convert context dict to structured string to feed to the AI
    # We include more data here for the doctor's expert review
    safe_context = {
        "patient_profile": patient_context.get("general_info", {}),
        "active_conditions": patient_context.get("conditions", []),
        "medications_and_adherence": {
            "current": patient_context.get("medications", []),
            "adherence_stats": patient_context.get("medication_adherence", {})
        },
        "allergies": patient_context.get("allergies", []),
        "vitals_history": patient_context.get("vitals", [])[-15:], # Last 15 readings for trend analysis
        "recent_clinical_alerts": patient_context.get("alerts", [])[:15], 
        "lab_metrics_trends": patient_context.get("lab_trends", {}),
        "respiratory_events_7d": patient_context.get("audio_summary", {}),
        "ai_generated_clinical_summary": patient_context.get("clinical_summary", "")
    }
    context_str = json.dumps(safe_context, indent=2, default=str)

    system_prompt = f"""You are a Senior Clinical AI Assistant powered by Gemini 2.5 Flash. 
You are assisting a licensed physician in analyzing a patient's case.

[STRICT PROTOCOL]:
1. ANSWERING: Use ONLY the provided [Patient Context] facts. 
2. REASONING: Connect dots between different data points (e.g., if BP is high and medication adherence is low, mention the correlation).
3. CLINICAL TONE: Be professional, objective, and precise. Use medical terminology correctly.
4. LIMITATIONS: If the data is insufficient for a conclusion, state: "The provided clinical data does not contain sufficient information to determine [X]."
5. HALLUCINATION: DO NOT invent vitals, lab results, or patient history.

[Patient Context]:
{context_str}
"""

    # Format history for Gemini API
    # chat_history is list of {"role": "user"|"model", "content": "..."}
    contents = []
    
    # Add previous turns
    for turn in chat_history:
        contents.append({
            "role": turn["role"],
            "parts": [{"text": turn["content"]}]
        })
    
    # Add current message
    # If no history, we include the system prompt/context in the first user message
    current_prompt = user_message
    if not chat_history:
        current_prompt = f"{system_prompt}\n\n[Doctor's Query]: {user_message}"
    else:
        # If there is history, we still remind it of the context briefly if needed, 
        # but Gemini usually holds the context well if it's in the first turn.
        # However, to be safe and ensure context is always fresh:
        current_prompt = f"[Note: Context Updated]\n{user_message}"

    contents.append({
        "role": "user",
        "parts": [{"text": current_prompt}]
    })
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            'contents': contents,
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.2, # Lower temperature for clinical accuracy
                "topP": 0.8
            }
        }
        
        print(f"[summarizer] Sending clinical chat request (turns: {len(contents)})")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                answer = data['candidates'][0]['content']['parts'][0]['text']
                return answer
        else:
            print(f"[summarizer] Gemini API error {resp.status_code}: {resp.text}")
        
        return "I apologize, but I am unable to analyze the clinical data at this moment."
    except Exception as e:
        print(f"[summarizer] AI Case Chat failed: {e}")
        return "An error occurred during clinical data analysis."
