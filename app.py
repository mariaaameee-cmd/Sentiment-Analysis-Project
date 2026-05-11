# -*- coding: utf-8 -*-
import os
import uuid
import sqlite3
import requests
import whisper
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template_string
from transformers import pipeline
from werkzeug.utils import secure_filename

# الإعدادات الأساسية - تم تغيير الموديلات لنسخ خفيفة جداً لتجنب الـ Crash
N8N_WEBHOOK_URL = "https://asmaamamdouh2005.app.n8n.cloud/webhook/crm-result"
TEXT_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion" # موديل خفيف جداً
WHISPER_MODEL_NAME = "tiny" # أصغر نسخة من ويسبير
UPLOAD_FOLDER = "uploads"
DB_PATH = "crm_sentiment.db"
MAX_CONTENT_LENGTH_MB = 25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm", "mp4", "mpeg"}

# تحميل الموديلات
print("Loading light-weight models...")
classifier = pipeline("sentiment-analysis", model=TEXT_MODEL_NAME)
speech_model = whisper.load_model(WHISPER_MODEL_NAME)
print("Models loaded successfully.")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            input_type TEXT NOT NULL,
            original_text TEXT,
            transcribed_text TEXT,
            sentiment TEXT NOT NULL,
            sentiment_ar TEXT NOT NULL,
            confidence REAL NOT NULL,
            recommended_action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def map_sentiment(label):
    label = label.lower()
    if label in ['joy', 'love', 'happy']:
        return "Happy", "سعيد"
    elif label in ['sadness', 'anger', 'fear', 'surprise']:
        return "Angry", "غاضب"
    else:
        return "Neutral", "محايد"

def recommended_action(sentiment):
    mapping = {
        "Angry": "Escalate immediately to customer support",
        "Happy": "Thank the customer and continue engagement",
        "Neutral": "Monitor conversation and collect more context",
    }
    return mapping.get(sentiment, "Manual review")

def analyze_text_message(text):
    result = classifier(text)[0]
    sentiment, sentiment_ar = map_sentiment(result.get("label"))
    return {
        "sentiment": sentiment,
        "sentiment_ar": sentiment_ar,
        "confidence": round(float(result.get("score", 0.0)), 4),
        "recommended_action": recommended_action(sentiment),
    }

def transcribe_audio(audio_path):
    result = speech_model.transcribe(audio_path, fp16=False)
    return (result.get("text") or "").strip()

def save_result(payload):
    conn = get_db_connection()
    cursor = conn.execute("""
        INSERT INTO customer_interactions
        (customer_name, input_type, original_text, transcribed_text, sentiment, sentiment_ar, confidence, recommended_action, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["customer_name"],
        payload["input_type"],
        payload.get("original_text"),
        payload.get("transcribed_text"),
        payload["sentiment"],
        payload["sentiment_ar"],
        payload["confidence"],
        payload["recommended_action"],
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id

def notify_n8n(payload):
    if not N8N_WEBHOOK_URL: return {"sent": False}
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        return {"sent": True, "status_code": response.status_code}
    except:
        return {"sent": False}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

HOME_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <title>CRM Sentiment Analysis</title>
  <style>
    body { font-family: sans-serif; background: #f5f7fb; text-align: center; padding: 20px; }
    .card { background: white; border-radius: 15px; padding: 20px; margin: 10px auto; max-width: 500px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    input, textarea, button { width: 90%; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
    button { background: #0b4f8a; color: white; cursor: pointer; font-weight: bold; }
    .result { background: #111827; color: #00ff00; padding: 15px; border-radius: 10px; text-align: left; margin-top: 20px; min-height: 50px; }
  </style>
</head>
<body>
    <h1>CRM Sentiment Analysis</h1>
    <div class="card">
        <h3>تحليل نص</h3>
        <input type="text" id="tName" placeholder="اسم العميل">
        <textarea id="tMsg" placeholder="اكتب الرسالة..."></textarea>
        <button onclick="analyzeText()">تحليل</button>
    </div>
    <div class="card">
        <h3>تحليل صوت</h3>
        <input type="text" id="vName" placeholder="اسم العميل">
        <input type="file" id="aFile" accept="audio/*">
        <button onclick="analyzeVoice()">تحليل</button>
    </div>
    <div class="result" id="res">النتيجة ستظهر هنا...</div>

    <script>
        async function analyzeText() {
            const name = document.getElementById("tName").value;
            const text = document.getElementById("tMsg").value;
            document.getElementById("res").innerText = "جاري التحليل...";
            const res = await fetch("/analyze_text", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({customer_name: name, text: text})
            });
            const data = await res.json();
            document.getElementById("res").innerText = JSON.stringify(data, null, 2);
        }

        async function analyzeVoice() {
            const name = document.getElementById("vName").value;
            const file = document.getElementById("aFile").files[0];
            const fd = new FormData();
            fd.append("customer_name", name);
            fd.append("audio", file);
            document.getElementById("res").innerText = "جاري رفع وتحليل الصوت...";
            const res = await fetch("/analyze_voice", { method: "POST", body: fd });
            const data = await res.json();
            document.getElementById("res").innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home(): return render_template_string(HOME_HTML)

@app.route("/analyze_text", methods=["POST"])
def analyze_text_api():
    data = request.get_json(silent=True) or request.form.to_dict()
    payload = {
        "customer_name": data.get("customer_name", "Unknown"),
        "input_type": "text",
        "original_text": data.get("text"),
        "transcribed_text": data.get("text"),
        **analyze_text_message(data.get("text")),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    record_id = save_result(payload)
    notify_n8n({**payload, "record_id": record_id})
    return jsonify(payload)

@app.route("/analyze_voice", methods=["POST"])
def analyze_voice_api():
    name = request.form.get("customer_name", "Unknown")
    file = request.files.get("audio")
    temp_path = f"temp_{uuid.uuid4().hex}.wav"
    file.save(temp_path)
    text = transcribe_audio(temp_path)
    os.remove(temp_path)
    payload = {
        "customer_name": name,
        "input_type": "voice",
        "transcribed_text": text,
        **analyze_text_message(text),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    record_id = save_result(payload)
    notify_n8n({**payload, "record_id": record_id})
    return jsonify(payload)

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)