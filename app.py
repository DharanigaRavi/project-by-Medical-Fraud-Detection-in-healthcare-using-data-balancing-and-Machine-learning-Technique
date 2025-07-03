from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from datetime import datetime
import joblib
import pandas as pd

import os

app = Flask(__name__)

app.secret_key = 'secretkey123'

# DB Setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Email Config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'daminmain@gmail.com'
app.config['MAIL_PASSWORD'] = 'kpqtxqskedcykwjz'
mail = Mail(app)

# Load model
model = joblib.load('fraud_detection_xgb_model.pkl')

# Category mappings
CATEGORY_MAPPINGS = {
    'gender': ['Female', 'Male'],
    'procedure_type': sorted(['Routine Check', 'Blood Test', 'X-Ray', 'Vaccination',
                             'MRI Scan', 'Surgery', 'Specialist Consult']),
    'diagnosis_code': sorted(['J45', 'I10', 'E11', 'M54', 'C34', 'I21', 'E66']),
    'claim_month': ['Jan', 'Mar', 'Jun', 'Dec']
}

FEATURE_ORDER = [
    'patient_age', 'gender', 'provider_id', 'hospital_id', 'claim_month',
    'procedure_type', 'diagnosis_code', 'claim_amount', 'num_procedures',
    'days_admitted', 'previous_claims', 'billing_discrepancy', 'anomaly_score',
    'month_fraud_risk', 'risk_score'
]


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    mail = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    age = db.Column(db.Integer)
    location = db.Column(db.String(100))
    extra1 = db.Column(db.String(100))
    extra2 = db.Column(db.String(100))


class ClaimResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    is_fraud = db.Column(db.Boolean, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Encode & Calculate Risk
def encode_category(feature_name, value):
    return CATEGORY_MAPPINGS[feature_name].index(value)

def calculate_risk_score(inputs):
    risk = (
        0.3 * inputs['anomaly_score'] +
        0.25 * inputs['billing_discrepancy'] +
        0.2 * (inputs['num_procedures'] / 10) +
        0.15 * (inputs['claim_amount'] / 100000) +
        0.1 * inputs['month_fraud_risk']
    )
    return min(max(risk, 0), 1)


# Send Email Alert
def send_fraud_alert_email(recipient, fraud_probability):
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

# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect('/analyze')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(
            name=request.form['name'],
            mail=request.form['mail'],
            password=request.form['password'],
            age=request.form['age'],
            location=request.form['location'],
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.")
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(mail=request.form['mail'], password=request.form['password']).first()
        if user:
            session['user_id'] = user.id
            session['user_mail'] = user.mail
            flash("Login successful.")
            return redirect('/analyze')
        flash("Invalid credentials.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.")
    return redirect('/login')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    result = None
    risk_factors = []

    if request.method == 'POST':
        inputs = {
            'patient_age': int(request.form['patient_age']),
            'gender': request.form['gender'],
            'provider_id': int(request.form['provider_id']),
            'hospital_id': int(request.form['hospital_id']),
            'procedure_type': request.form['procedure_type'],
            'diagnosis_code': request.form['diagnosis_code'],
            'claim_amount': float(request.form['claim_amount']),
            'num_procedures': int(request.form['num_procedures']),
            'days_admitted': int(request.form['days_admitted']),
            'previous_claims': int(request.form['previous_claims']),
            'billing_discrepancy': float(request.form['billing_discrepancy']),
            'anomaly_score': float(request.form['anomaly_score']),
            'claim_month': request.form['claim_month']
        }

        inputs['month_fraud_risk'] = {'Jan': 0.25, 'Mar': 0.22, 'Jun': 0.18, 'Dec': 0.35}[inputs['claim_month']]
        inputs['risk_score'] = calculate_risk_score(inputs)

        encoded_inputs = {k: encode_category(k, v) if k in CATEGORY_MAPPINGS else v for k, v in inputs.items()}
        claim_data = pd.DataFrame([encoded_inputs])[FEATURE_ORDER]

        prediction = model.predict(claim_data)[0]
        probability = model.predict_proba(claim_data)[0][1]
        status = "Fraud Detected ⚠️" if prediction == 1 else "Legitimate Claim ✅"

        if inputs['hospital_id'] == 99:
            risk_factors.append("⚠️ High-risk hospital (ID 99)")
        if inputs['risk_score'] > 0.7:
            risk_factors.append(f"⚠️ Extreme risk score ({inputs['risk_score']:.2f})")
        if inputs['anomaly_score'] > 0.8:
            risk_factors.append(f"⚠️ High anomaly score ({inputs['anomaly_score']:.2f})")

        result = {
            'probability': round(probability * 100, 1),
            'status': status,
            'risk_factors': risk_factors,
            'feature_values': claim_data.to_dict(orient='records')[0]
        }

        # Save to DB
        claim_entry = ClaimResult(
            user_email=session.get('user_mail'),
            probability=result['probability'],
            is_fraud=(prediction == 1),
            risk_score=inputs['risk_score']
        )
        db.session.add(claim_entry)
        db.session.commit()

        # Send email if fraud detected
        if prediction == 1:
            send_fraud_alert_email(session['user_mail'], result['probability'])

    return render_template('index.html', mappings=CATEGORY_MAPPINGS, result=result)

@app.route('/history')
def history():
    if 'user_mail' not in session:
        flash("Please login to view your history.")
        return redirect('/login')

    user_email = session['user_mail']
    history_data = ClaimResult.query.filter_by(user_email=user_email).order_by(ClaimResult.timestamp.desc()).all()

    return render_template('history.html', history=history_data)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
