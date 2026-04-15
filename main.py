from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import difflib
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

# ================= LOAD DATA =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Data Model2.csv")

df = pd.read_csv(file_path)

df.fillna("", inplace=True)
df.replace("-", "", inplace=True)

# Normalize
df["subject"] = df["subject"].str.lower()
df["FrameWork"] = df["FrameWork"].str.lower()
df["Language"] = df["Language"].str.lower()
df["level"] = df["level"].str.lower()

subjects = df["subject"].unique()
frameworks = df["FrameWork"].unique()
languages = df["Language"].unique()

# ================= REQUEST MODEL =================
class UserInput(BaseModel):
    message: str

# ================= ALIASES =================
subject_aliases = {
    "web / frontend": ["web", "frontend", "front", "ui"],
    "backend": ["backend", "back", "api", "server"],
    "data science": ["ds", "data science"],
    "data analysis": ["analysis", "data analysis"],
    "ai / artificial intelligence": ["ai", "artificial intelligence"],
}

framework_aliases = {
    "machinelearning": ["ml", "machine learning", "machine"],
    "deeplearning": ["dl", "deep learning", "deep"],
    "react": ["react", "reactjs"],
    "nodejs": ["node", "nodejs"],
    "python": ["python", "py"],
}

# ================= DETECTION =================
def detect_subject(text):
    text = text.lower()

    for subject, aliases in subject_aliases.items():
        for a in aliases:
            if a in text:
                return subject

    for s in subjects:
        if s in text:
            return s

    return None


def detect_framework(text):
    text = text.lower()

    for fw, aliases in framework_aliases.items():
        for a in aliases:
            if a in text:
                return fw

    for f in frameworks:
        if f in text:
            return f

    return None


def detect_language(text):
    text = text.lower()

    for lang in languages:
        if lang in text:
            return lang

    return None


# ================= RECOMMENDER =================
def recommend(subject=None, framework=None, language=None):
    results = df.copy()

    if subject:
        results = results[results["subject"] == subject]

    if framework:
        results = results[results["FrameWork"] == framework]

    if language:
        results = results[results["Language"] == language]

    if results.empty:
        return []

    results = results.sample(frac=1)
    return results.head(5).to_dict(orient="records")


# ================= API =================
@app.post("/chat")
def chat(user: UserInput):
    text = user.message.lower()

    subject = detect_subject(text)
    framework = detect_framework(text)
    language = detect_language(text)

    if not subject and framework:
        # infer subject
        row = df[df["FrameWork"] == framework]
        if not row.empty:
            subject = row.iloc[0]["subject"]

    courses = recommend(subject, framework, language)

    return {
        "detected": {
            "subject": subject,
            "framework": framework,
            "language": language
        },
        "courses": courses if courses else "No courses found"
    }