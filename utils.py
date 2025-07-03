# utils.py

from flask_mail import Message,mail
from app import app

def send_fraud_alert_email(recipient, fraud_probability):
    with app.app_context():
        msg = Message(
            subject="⚠️ Fraud Alert Notification",
            sender=app.config['MAIL_USERNAME'],
            recipients=[recipient],
            body=(
                f"Dear User,\n\n"
                f"A suspicious claim was flagged with a fraud probability of {fraud_probability:.2f}%.\n"
                "Please verify the claim for further investigation.\n\n"
                "Thanks,\nFraud Detection System"
            )
        )
        mail.send(msg)
