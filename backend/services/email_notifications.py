"""Email Notification Service for CareTaker.

Sends:
- Medicine reminder emails at scheduled times
- Report upload reminders (no report for 30+ days)
- Medication completion notifications
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(override=True)


def _get_smtp_config():
    """Get SMTP configuration from environment."""
    return {
        "server": os.getenv("MAIL_SERVER", "smtp.gmail.com"),
        "port": int(os.getenv("MAIL_PORT", 587)),
        "username": os.getenv("MAIL_USERNAME"),
        "password": os.getenv("MAIL_PASSWORD"),
        "from_email": os.getenv("MAIL_FROM"),
        "starttls": os.getenv("MAIL_STARTTLS", "True").lower() == "true",
    }


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    config = _get_smtp_config()
    if not config["username"] or not config["password"]:
        print("[email] SMTP credentials not configured, skipping email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = config["from_email"]
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(config["server"], config["port"], timeout=30) as server:
            if config["starttls"]:
                server.starttls()
            server.login(config["username"], config["password"])
            server.sendmail(config["from_email"], to_email, msg.as_string())

        print(f"[email] ✅ Sent to {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"[email] ❌ Failed to send to {to_email}: {e}")
        return False


def send_medicine_reminder(to_email: str, recipient_name: str, medicine_name: str, dosage: str, schedule_time: str):
    """Send a medication reminder email."""
    subject = f"💊 Medicine Reminder: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">💊</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medicine Reminder</h2>
        </div>
        <div style="background: linear-gradient(135deg, #ebf8ff, #e6fffa); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Dosage:</strong> {dosage or 'As prescribed'}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Scheduled Time:</strong> {schedule_time}</p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            This is an automated reminder from CareTaker AI.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_medication_completed(to_email: str, recipient_name: str, medicine_name: str, duration_days: int):
    """Send a notification that a medication course is complete."""
    subject = f"✅ Medication Completed: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">✅</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medication Course Completed</h2>
        </div>
        <div style="background: linear-gradient(135deg, #f0fff4, #e6fffa); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Duration:</strong> {duration_days} days</p>
            <p style="margin: 8px 0; color: #38a169; font-weight: 600;">
                This medication has been automatically marked as completed.
            </p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            Consult your doctor before making any changes to the medication plan.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_report_upload_reminder(to_email: str, recipient_name: str, days_since_last: int):
    """Send a reminder that no medical report has been uploaded recently."""
    subject = f"📋 Report Reminder: No new reports for {recipient_name} in {days_since_last} days"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">📋</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Medical Report Reminder</h2>
        </div>
        <div style="background: linear-gradient(135deg, #fffaf0, #fefcbf); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 8px 0; color: #c05621; font-weight: 600;">
                ⚠️ It has been {days_since_last} days since the last medical report was uploaded.
            </p>
            <p style="margin: 4px 0; color: #4a5568;">
                Regular medical reports help track health trends and detect issues early.
                Please upload a recent report to keep the health profile up to date.
            </p>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            This is an automated reminder from CareTaker AI.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


def send_auto_reorder_notification(to_email: str, recipient_name: str, medicine_name: str, dosage: str, current_stock: int, days_remaining: int, order_link: str):
    """Send a notification that an auto-reorder has been triggered via Tata 1mg."""
    subject = f"🛒 Auto-Reorder Triggered: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">🛒</span>
            <h2 style="color: #2d3748; margin: 8px 0;">Auto-Reorder Alert</h2>
        </div>
        <div style="background: linear-gradient(135deg, #fff5f5, #fed7d7); padding: 20px; border-radius: 8px; margin-bottom: 16px;">
            <p style="margin: 4px 0; color: #4a5568;"><strong>Patient:</strong> {recipient_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Medicine:</strong> {medicine_name}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Dosage:</strong> {dosage or 'As prescribed'}</p>
            <p style="margin: 4px 0; color: #4a5568;"><strong>Current Stock:</strong> {current_stock} units</p>
            <p style="margin: 8px 0; color: #c53030; font-weight: 600;">
                ⚠️ Only {days_remaining} day(s) of stock remaining!
            </p>
        </div>
        <div style="text-align: center; margin: 20px 0;">
            <a href="{order_link}" 
               style="display: inline-block; background: linear-gradient(135deg, #FF6B35, #F72C25); color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 1rem;">
                🛍️ Order Now on Tata 1mg
            </a>
        </div>
        <p style="color: #718096; font-size: 13px; text-align: center;">
            CareTaker AI has detected low stock and generated this order link automatically.<br>
            Click the button above to review and complete the purchase on Tata 1mg.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Dose Confirmation Email  (actionable — caretaker clicks TAKEN / MISSED)
# ─────────────────────────────────────────────────────────────────────────────

def send_dose_confirmation_email(
    to_email: str,
    recipient_name: str,
    medicine_name: str,
    dosage: str,
    schedule_time: str,
    dose_log_id: int,
    token: str,
) -> bool:
    """
    Send a medicine confirmation email with two actionable buttons:
      - ✔ Mark as TAKEN  → GET /api/confirm-dose-email?token=TOKEN&action=TAKEN
      - ✖ Mark as NOT TAKEN → GET /api/confirm-dose-email?token=TOKEN&action=MISSED

    The token is a UUID4 tied to this specific dose log.  Both links are safe
    to embed in email clients because they use plain GET requests.
    """
    import os
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    taken_url  = f"{base_url}/api/confirm-dose-email?token={token}&action=TAKEN"
    missed_url = f"{base_url}/api/confirm-dose-email?token={token}&action=MISSED"

    subject = f"💊 Confirm Medicine: {medicine_name} for {recipient_name}"
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;padding:28px;border-radius:16px;border:1px solid #e2e8f0;background:#fff;">

        <!-- Header -->
        <div style="text-align:center;margin-bottom:24px;">
            <span style="font-size:52px;">💊</span>
            <h2 style="color:#1e3a8a;margin:10px 0 4px;font-size:1.4rem;">Medicine Confirmation Required</h2>
            <p style="color:#64748b;margin:0;font-size:0.9rem;">Please confirm whether the dose was taken</p>
        </div>

        <!-- Info Card -->
        <div style="background:linear-gradient(135deg,#eff6ff,#e0f2fe);padding:20px;border-radius:12px;margin-bottom:24px;">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;width:120px;"><strong>Patient</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{recipient_name}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Medicine</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:700;">{medicine_name}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Dosage</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{dosage or 'As prescribed'}</td></tr>
                <tr><td style="padding:4px 0;color:#64748b;font-size:0.9rem;"><strong>Scheduled</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{schedule_time}</td></tr>
            </table>
        </div>

        <!-- Action Buttons -->
        <div style="display:flex;gap:12px;justify-content:center;margin-bottom:20px;">
            <a href="{taken_url}"
               style="display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:white;
                      padding:14px 28px;border-radius:10px;text-decoration:none;font-weight:700;
                      font-size:1rem;text-align:center;min-width:160px;">
                ✔&nbsp; Mark as TAKEN
            </a>
            <a href="{missed_url}"
               style="display:inline-block;background:#fff;color:#ef4444;border:2px solid #ef4444;
                      padding:12px 28px;border-radius:10px;text-decoration:none;font-weight:700;
                      font-size:1rem;text-align:center;min-width:160px;">
                ✖&nbsp; Not Taken
            </a>
        </div>

        <!-- Footer -->
        <p style="color:#94a3b8;font-size:0.78rem;text-align:center;margin:0;">
            This link expires after 24 hours.<br>
            Do not forward this email — the confirmation link is unique to this dose.
        </p>
    </div>
    """
    return send_email(to_email, subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Missed Dose Escalation Email  (urgent alert to caretaker)
# ─────────────────────────────────────────────────────────────────────────────

def send_missed_dose_escalation(
    to_email: str,
    recipient_name: str,
    medicine_name: str,
    dosage: str,
    scheduled_time: str,
) -> bool:
    """
    Send an urgent escalation email when the system auto-marks a dose as MISSED
    (no confirmation received within 60 minutes of the scheduled time).
    """
    subject = f"🚨 MISSED DOSE ALERT: {medicine_name} — {recipient_name}"
    html = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;padding:28px;border-radius:16px;border:2px solid #ef4444;background:#fff;">

        <!-- Urgent Header -->
        <div style="background:linear-gradient(135deg,#ef4444,#dc2626);border-radius:12px;padding:20px;text-align:center;margin-bottom:24px;">
            <span style="font-size:48px;">🚨</span>
            <h2 style="color:white;margin:8px 0 4px;font-size:1.4rem;">Missed Dose Alert</h2>
            <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.9rem;">Immediate attention required</p>
        </div>

        <!-- Info Card -->
        <div style="background:#fff5f5;padding:20px;border-radius:12px;border:1px solid #fecaca;margin-bottom:20px;">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;width:120px;"><strong>Patient</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:600;">{recipient_name}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Medicine</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;font-weight:700;">{medicine_name}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Dosage</strong></td>
                    <td style="color:#0f172a;font-size:0.9rem;">{dosage or 'As prescribed'}</td></tr>
                <tr><td style="padding:5px 0;color:#64748b;font-size:0.9rem;"><strong>Scheduled</strong></td>
                    <td style="color:#c53030;font-size:0.9rem;font-weight:700;">{scheduled_time}</td></tr>
            </table>
        </div>

        <!-- Body Message -->
        <p style="color:#1e293b;font-size:1rem;font-weight:600;margin-bottom:8px;">
            ⚠️ {recipient_name} has <span style="color:#ef4444;">not confirmed</span> taking
            <strong>{medicine_name}</strong> scheduled at <strong>{scheduled_time}</strong>.
        </p>
        <p style="color:#475569;font-size:0.9rem;margin-bottom:20px;">
            This dose has been automatically marked as <strong>MISSED</strong>.
            Please contact the patient immediately and verify their status.
        </p>

        <!-- Footer -->
        <p style="color:#94a3b8;font-size:0.78rem;text-align:center;margin:0;">
            CareTaker AI — Automated Escalation Alert
        </p>
    </div>
    """
    return send_email(to_email, subject, html)

