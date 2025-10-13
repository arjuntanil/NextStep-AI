from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import shutil
import uuid
import json
from resume_parser import parse_resume
import numpy as np
import re
from typing import Optional

def normalize_skill(skill):
    skill = skill.lower().strip()
    skill = re.sub(r"( programming| language| developer| engineer| basics| advanced| skills| knowledge| experience)", "", skill)
    return skill.strip()

MODEL_PATH = 'skill_predictor.joblib'
VECTORIZER_PATH = 'job_title_vectorizer.joblib'
MLB_PATH = 'skill_mlb.joblib'
HISTORY_PATH = 'history.json'
UPLOAD_DIR = 'uploads'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
mlb = joblib.load(MLB_PATH)

os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

@app.post('/upload_resume')
def upload_resume(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ['.pdf', '.docx']:
        return JSONResponse(status_code=400, content={"error": "Only PDF and DOCX files are supported."})
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    skills = parse_resume(file_path)
    # Normalize extracted skills
    skills = [normalize_skill(s) for s in skills]
    return {"file_id": file_id, "skills": skills}

@app.post('/predict_skills')
def predict_skills(job_title: str = Form(...)):
    X = vectorizer.transform([job_title.strip().lower()])
    y_proba = clf.predict_proba(X)
    # Use a threshold to select predicted skills
    threshold = 0.3
    y_pred = (y_proba[0] >= threshold).astype(int)
    skills = mlb.inverse_transform([y_pred])[0]
    return {"predicted_skills": list(skills)}

@app.post('/skill_gap')
def skill_gap(resume_skills: str = Form(...), job_title: str = Form(...)):
    # Split the comma-separated string into a list, handle empty input
    if resume_skills.strip():
        resume_skills_list = [s.strip() for s in resume_skills.split(',') if s.strip()]
    else:
        resume_skills_list = []
    X = vectorizer.transform([job_title.strip().lower()])
    y_proba = clf.predict_proba(X)
    threshold = 0.3
    y_pred = (y_proba[0] >= threshold).astype(int)
    predicted_skills = set(mlb.inverse_transform([y_pred])[0])
    resume_skills_set = set([normalize_skill(s) for s in resume_skills_list])
    gap = list(predicted_skills - resume_skills_set)
    return {"skill_gap": gap, "predicted_skills": list(predicted_skills)}

@app.post('/save_history')
def save_user_history(file_id: str = Form(...), job_title: str = Form(...), resume_skills: str = Form(...), predicted_skills: str = Form(...), skill_gap: str = Form(...)):
    history = load_history()
    entry = {
        "file_id": file_id,
        "job_title": job_title,
        "resume_skills": resume_skills,
        "predicted_skills": predicted_skills,
        "skill_gap": skill_gap
    }
    history.append(entry)
    save_history(history)
    return {"status": "success"}

@app.get('/history')
def get_history():
    history = load_history()
    return {"history": history} 