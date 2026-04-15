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


def send_urgent_doctor_alert(to_email: str, recipient_name: str, urgent_flags: list, trend_summary: str):
    """Notify the doctor immediately about serious health changes."""
    subject = f"🚨 URGENT: Health Alert for {recipient_name}"
    
    flags_html = "".join([f"<li style='color: #c53030;'><strong>{f}</strong></li>" for f in urgent_flags])
    
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 2px solid #feb2b2;">
        <div style="background-color: #fff5f5; padding: 16px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #f56565;">
            <h2 style="color: #c53030; margin: 0;">⚠️ Urgent Medical Alert</h2>
            <p style="color: #742a2a; margin: 8px 0 0 0;">CareTaker AI has detected serious health changes for: <strong>{recipient_name}</strong></p>
        </div>
        
        <div style="margin-bottom: 24px;">
            <h3 style="color: #2d3748; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">Serious Findings:</h3>
            <ul>{flags_html}</ul>
        </div>
        
        <div style="background-color: #f7fafc; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
            <h3 style="color: #2d3748; margin-top: 0;">AI Analysis:</h3>
            <p style="color: #4a5568; line-height: 1.6;">{trend_summary}</p>
        </div>
        
        <div style="text-align: center; margin-top: 32px;">
            <p style="color: #718096; font-size: 14px; margin-bottom: 16px;">Please review the patient's updated clinical dashboard for full vitals and lab history.</p>
            <a href="http://127.0.0.1:8000/api/doctor/patients" 
               style="display: inline-block; background-color: #3182ce; color: white; padding: 12px 24px; border-radius: 6px; text-decoration: none; font-weight: 600;">
               Review Clinical Dashboard
            </a>
        </div>
    </div>
    """
    return send_email(to_email, subject, html)


def send_periodic_doctor_summary(to_email: str, recipient_name: str, summary_html: str):
    """Send a 3-day periodic health summary to the doctor."""
    subject = f"📋 Health Summary (3-Day): {recipient_name}"
    
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="text-align: center; margin-bottom: 24px;">
            <h2 style="color: #2d3748; margin: 0;">Periodic Clinical Summary</h2>
            <p style="color: #718096;">Patient: <strong>{recipient_name}</strong></p>
        </div>
        
        <div style="margin-top: 24px;">
            {summary_html}
        </div>
        
        <div style="margin-top: 32px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center;">
            <p style="color: #a0aec0; font-size: 12px;">This is an automated 3-day health summary from CareTaker AI Proactive Health Intelligence.</p>
        </div>
    </div>
    """
    return send_email(to_email, subject, html)
