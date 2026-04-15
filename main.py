from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib

app = FastAPI()

# ✅ CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting API...")

# ✅ Load dataset ONCE
df = pd.read_csv("Data Model2.csv")
df.fillna("", inplace=True)

print("✅ Dataset loaded")

# ==============================
# DATA PREPARATION
# ==============================

subjects = df["subject"].str.lower().unique()
frameworks = df["FrameWork"].str.lower().unique()
languages = df["Language"].str.lower().unique()

all_keywords = list(subjects) + list(frameworks)

# ==============================
# ALIASES
# ==============================

subject_aliases = {
    "web / frontend": ["web", "frontend", "front", "ui"],
    "backend": ["backend", "back", "api", "server"],
    "data science": ["data science", "ds"],
    "data analysis": ["data analysis", "analysis"],
    "ai / artificial intelligence": ["ai", "artificial intelligence"],
}

framework_aliases = {
    "machinelearning": ["ml", "machine learning", "machine"],
    "deeplearning": ["dl", "deep learning", "deep"],
    "react": ["react", "reactjs"],
    "nodejs": ["node", "nodejs"],
    "python": ["python", "py"],
}

language_aliases = {
    "javascript": ["js", "javascript"],
    "html": ["html"],
    "css": ["css"],
    "java": ["java"],
}

# ==============================
# DETECTION FUNCTIONS
# ==============================

def correct_word(word):
    matches = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.8)
    return matches[0] if matches else None

def detect_subject(text):
    text = text.lower()

    for subject, aliases in subject_aliases.items():
        for alias in aliases:
            if alias in text:
                return subject

    for s in subjects:
        if s in text:
            return s

    return None

def detect_framework(text):
    text = text.lower()

    for fw, aliases in framework_aliases.items():
        for alias in aliases:
            if alias in text:
                return fw

    for fw in frameworks:
        if fw in text:
            return fw

    return None

def detect_language(text):
    text = text.lower()

    for lang, aliases in language_aliases.items():
        for alias in aliases:
            if alias in text:
                return lang

    for lang in languages:
        if lang in text:
            return lang

    return None

# ==============================
# RECOMMENDATION
# ==============================

def recommend_courses(subject=None, framework=None, language=None):
    results = df.copy()

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    if len(results) == 0:
        return []

    results = results.sample(frac=1).head(5)

    output = []
    for _, row in results.iterrows():
        output.append({
            "title": row["course_title"],
            "subject": row["subject"],
            "framework": row["FrameWork"],
            "language": row["Language"],
            "level": row["level"],
            "url": row["url"]
        })

    return output

# ==============================
# API ROUTES
# ==============================

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.get("/chat")
def chat(query: str):
    subject = detect_subject(query)
    framework = detect_framework(query)
    language = detect_language(query)

    results = recommend_courses(subject, framework, language)

    return {
        "detected": {
            "subject": subject,
            "framework": framework,
            "language": language
        },
        "courses": results
    }