"""
Background Notification Scheduler for CareTaker.

Runs periodic tasks:
1. Medicine reminders    — checks active meds every minute, sends email at scheduled times
2. Medication expiry     — marks completed if duration_days has passed, sends completion email
3. Report reminders      — weekly check, emails caretaker if no report for 30+ days
4. Daily Recommendations — re-runs clinical logic for all recipients every 24h
"""

import threading
import time
import datetime
from typing import Optional

_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

# Track sent reminders to avoid duplicates (resets on restart — OK for dev)
# key: "{med_id}_{YYYY-MM-DD}_{HH}" → bool
_reminders_sent: dict = {}


def _get_db():
    from config import SessionLocal
    return SessionLocal()


def check_medicine_reminders():
    """
    For every active medication with a schedule_time, check if the current
    time matches any of the scheduled times (within a 1-minute window) and
    send a reminder email if not already sent today for that slot.
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.users import CareRecipient, CareTaker

        now = datetime.datetime.now()
        current_hhmm = now.strftime("%H:%M")
        today_str = now.strftime("%Y-%m-%d")

        active_meds = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.schedule_time.isnot(None)
        ).all()

        for med in active_meds:
            # schedule_time may be comma-separated: "08:00,14:00,20:00"
            times = [t.strip() for t in med.schedule_time.split(",") if t.strip()]
            for slot in times:
                # Normalize slot to HH:MM (e.g. "1:15" -> "01:15")
                try:
                    slot_normalized = datetime.datetime.strptime(slot, "%H:%M").strftime("%H:%M")
                except ValueError:
                    # Fallback for formats like "1:15" if strptime fails (though 1:15 works with %H:%M on some systems, 
                    # we ensure consistency by rebuilding it)
                    try:
                        h, m = slot.split(':')
                        slot_normalized = f"{int(h):02d}:{int(m):02d}"
                    except Exception:
                        slot_normalized = slot

                reminder_key = f"{med.medication_id}_{today_str}_{slot_normalized}"
                if _reminders_sent.get(reminder_key):
                    continue  # Already sent today for this slot
                
                if slot_normalized == current_hhmm:
                    # Get recipient + caretaker email
                    recipient = db.query(CareRecipient).filter(
                        CareRecipient.id == med.care_recipient_id
                    ).first()
                    if not recipient:
                        continue
                    caretaker = db.query(CareTaker).filter(
                        CareTaker.id == recipient.caretaker_id
                    ).first()
                    if not caretaker or not caretaker.email:
                        continue

                    from services.email_notifications import send_medicine_reminder
                    sent = send_medicine_reminder(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name,
                        medicine_name=med.medicine_name,
                        dosage=med.dosage or "As prescribed",
                        schedule_time=slot
                    )
                    if sent:
                        _reminders_sent[reminder_key] = True
                        print(f"[scheduler] Medicine reminder sent: {med.medicine_name} at {slot}")
    except Exception as e:
        print(f"[scheduler] Medicine reminder error: {e}")
    finally:
        db.close()


def check_medication_expiry():
    """
    Check if any active medication has passed its end_date.
    If so, mark it 'completed' and send a completion email.
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.users import CareRecipient, CareTaker

        today = datetime.date.today()

        expired = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.end_date.isnot(None),
            Medication.end_date < today
        ).all()

        for med in expired:
            med.status = MedicationStatus.completed
            db.flush()

            recipient = db.query(CareRecipient).filter(
                CareRecipient.id == med.care_recipient_id
            ).first()
            if not recipient:
                continue
            caretaker = db.query(CareTaker).filter(
                CareTaker.id == recipient.caretaker_id
            ).first()
            if caretaker and caretaker.email:
                from services.email_notifications import send_medication_completed
                send_medication_completed(
                    to_email=caretaker.email,
                    recipient_name=recipient.full_name,
                    medicine_name=med.medicine_name,
                    duration_days=(med.end_date - med.start_date).days if med.end_date and med.start_date else 0
                )
            print(f"[scheduler] [OK] Medication '{med.medicine_name}' auto-completed (expired {med.end_date})")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[scheduler] Medication expiry check error: {e}")
    finally:
        db.close()


def check_report_reminders():
    """
    For every care recipient, check when the last medical report was uploaded.
    If it was more than 30 days ago (or never), send an email reminder.
    Only sends once per week per recipient by tracking in _reminders_sent.
    """
    db = _get_db()
    try:
        from tables.users import CareRecipient, CareTaker
        from tables.medical_reports import MedicalReport
        from sqlalchemy import func

        today = datetime.date.today()
        week_key = today.strftime("%Y-W%W")  # e.g. "2026-W10"

        recipients = db.query(CareRecipient).all()
        for recipient in recipients:
            reminder_key = f"report_{recipient.id}_{week_key}"
            if _reminders_sent.get(reminder_key):
                continue  # Already reminded this week

            # Find most recent report
            latest = db.query(func.max(MedicalReport.report_date)).filter(
                MedicalReport.care_recipient_id == recipient.id
            ).scalar()

            if latest is None:
                days_since = 999  # No reports ever
            elif isinstance(latest, datetime.datetime):
                days_since = (today - latest.date()).days
            else:
                days_since = (today - latest).days

            if days_since >= 30:
                caretaker = db.query(CareTaker).filter(
                    CareTaker.id == recipient.caretaker_id
                ).first()
                if caretaker and caretaker.email:
                    from services.email_notifications import send_report_upload_reminder
                    sent = send_report_upload_reminder(
                        to_email=caretaker.email,
                        recipient_name=recipient.full_name,
                        days_since_last=days_since if days_since < 999 else 30
                    )
                    if sent:
                        _reminders_sent[reminder_key] = True
                        print(f"[scheduler] Report reminder sent for {recipient.full_name} ({days_since} days)")
    except Exception as e:
        print(f"[scheduler] Report reminder error: {e}")
    finally:
        db.close()


def decrement_daily_stock():
    """
    Once per day, subtract doses_per_day from current_stock for every active
    medication.  Stock is clamped to 0 (never negative).
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus

        active_meds = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.current_stock > 0
        ).all()

        for med in active_meds:
            daily = med.doses_per_day or 1
            old_stock = med.current_stock or 0
            new_stock = max(0, old_stock - daily)
            med.current_stock = new_stock
            print(f"[scheduler] Stock decrement: {med.medicine_name} {old_stock} → {new_stock}")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[scheduler] Stock decrement error: {e}")
    finally:
        db.close()


def check_auto_reorder():
    """
    For every active medication with auto_order_enabled, check if stock is
    ≤ 7 days from running out.  If so, send a Tata 1mg order link email to
    the caretaker (once per 7-day window to avoid spam).
    """
    db = _get_db()
    try:
        from tables.medications import Medication, MedicationStatus
        from tables.users import CareRecipient, CareTaker
        from urllib.parse import quote

        today = datetime.date.today()

        meds = db.query(Medication).filter(
            Medication.status == MedicationStatus.active,
            Medication.auto_order_enabled == True
        ).all()

        for med in meds:
            daily = med.doses_per_day or 1
            stock = med.current_stock or 0
            days_remaining = stock // daily if daily > 0 else 999

            if days_remaining > 7:
                continue  # Enough stock

            # Prevent duplicate orders within 7 days
            if med.last_auto_order_date:
                days_since_last_order = (today - med.last_auto_order_date).days
                if days_since_last_order < 7:
                    continue

            # Build Tata 1mg link
            order_link = f"https://www.1mg.com/search/all?name={quote(med.medicine_name)}"

            # Get caretaker email
            recipient = db.query(CareRecipient).filter(
                CareRecipient.id == med.care_recipient_id
            ).first()
            if not recipient:
                continue
            caretaker = db.query(CareTaker).filter(
                CareTaker.id == recipient.caretaker_id
            ).first()
            if not caretaker or not caretaker.email:
                continue

            from services.email_notifications import send_auto_reorder_notification
            sent = send_auto_reorder_notification(
                to_email=caretaker.email,
                recipient_name=recipient.full_name,
                medicine_name=med.medicine_name,
                dosage=med.dosage or "As prescribed",
                current_stock=stock,
                days_remaining=days_remaining,
                order_link=order_link
            )

            # Update last_auto_order_date regardless of email success (avoid spam)
            med.last_auto_order_date = today
            db.flush()

            status_str = "[OK] email sent" if sent else "[WARN] email failed (link still generated)"
            print(f"[scheduler] [REORDER] Auto-reorder triggered: {med.medicine_name} "
                  f"({days_remaining}d left) -> Tata 1mg -- {status_str}")

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[scheduler] Auto-reorder check error: {e}")
    finally:
        db.close()


def run_daily_recommendations():
    """
    Reruns the clinical recommendation engine for every care recipient.
    Ensures that even without new reports, recommendations adapt to 
    changing vitals and audio events.
    """
    db = _get_db()
    try:
        from tables.users import CareRecipient
        from services.recommendation_engine import generate_recommendations
        
        recipients = db.query(CareRecipient).all()
        for recipient in recipients:
            try:
                generate_recommendations(recipient.id, db)
                print(f"[scheduler] Regenerated recommendations for {recipient.full_name} (ID: {recipient.id})")
            except Exception as e:
                print(f"[scheduler] Failed recommendations for recipient {recipient.id}: {e}")
    except Exception as e:
        print(f"[scheduler] Daily recommendations task failed: {e}")
    finally:
        db.close()


def _scheduler_loop():
    """Main scheduler loop — runs every 60 seconds."""
    print("[scheduler] [START] Background notification scheduler started")

    # Track which daily tasks last ran
    last_expiry_check = None
    last_report_check = None
    last_stock_decrement = None
    last_auto_reorder_check = None
    last_recommendation_run = None

    while not _stop_event.is_set():
        now = datetime.datetime.now()
        today = now.date()

        # Medicine reminders — check every minute
        try:
            check_medicine_reminders()
        except Exception as e:
            print(f"[scheduler] Reminder loop error: {e}")

        # Medication expiry — once per day
        if last_expiry_check != today:
            try:
                check_medication_expiry()
                last_expiry_check = today
            except Exception as e:
                print(f"[scheduler] Expiry loop error: {e}")

        # Report reminders — once per day
        if last_report_check != today:
            try:
                check_report_reminders()
                last_report_check = today
            except Exception as e:
                print(f"[scheduler] Report reminder loop error: {e}")

        # Stock decrement — once per day
        if last_stock_decrement != today:
            try:
                decrement_daily_stock()
                last_stock_decrement = today
            except Exception as e:
                print(f"[scheduler] Stock decrement loop error: {e}")

        # Auto-reorder check — once per day (after stock decrement)
        if last_auto_reorder_check != today:
            try:
                check_auto_reorder()
                last_auto_reorder_check = today
            except Exception as e:
                print(f"[scheduler] Auto-reorder loop error: {e}")

        # Daily Recommendations — once per day
        if last_recommendation_run != today:
            try:
                run_daily_recommendations()
                last_recommendation_run = today
            except Exception as e:
                print(f"[scheduler] Daily recommendations loop error: {e}")

        # Sleep 60 seconds (checking stop every second for clean shutdown)
        for _ in range(60):
            if _stop_event.is_set():
                break
            time.sleep(1)

    print("[scheduler] [STOP] Notification scheduler stopped")


def start_scheduler():
    """Start the background scheduler thread (idempotent)."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        print("[scheduler] Already running")
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="NotificationScheduler")
    _scheduler_thread.start()


def stop_scheduler():
    """Signal the scheduler to stop."""
    _stop_event.set()

