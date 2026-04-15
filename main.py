from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib
import os
import uvicorn

# =========================
# CREATE APP FIRST
# =========================
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting API...")

# =========================
# LOAD DATASET
# =========================
@app.on_event("startup")
def load_data():
    global df
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "Data Model2.csv")

    print("📂 Loading dataset...")
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    print("✅ Dataset loaded")

print("✅ Dataset loaded")

# =========================
# DATA PREP
# =========================
subjects = df["subject"].str.lower().unique()
frameworks = df["FrameWork"].str.lower().unique()
languages = df["Language"].str.lower().unique()

all_keywords = list(subjects) + list(frameworks)

# =========================
# ALIASES (IMPROVED)
# =========================
subject_aliases = {
    "web / frontend": ["web", "frontend", "front", "ui", "website"],
    "backend": ["backend", "back", "api", "server"],
    "data science": ["data science", "ds"],
    "data analysis": ["data analysis", "analysis"],
    "ai / artificial intelligence": ["ai", "artificial intelligence"],
}

framework_aliases = {
    "machinelearning": ["ml", "machine learning", "machine"],
    "deeplearning": ["dl", "deep learning", "deep"],
    "react": ["react", "reactjs"],
    "nodejs": ["node", "nodejs", "node js"],
    "python": ["python", "py"],
}

language_aliases = {
    "javascript": ["js", "javascript"],
    "html": ["html"],
    "css": ["css"],
    "java": ["java"],
}

# =========================
# DETECTION FUNCTIONS (IMPROVED)
# =========================
def detect_subject(text):
    text = text.lower()

    # 🔥 priority keywords
    if any(w in text for w in ["web", "frontend", "front"]):
        return "web / frontend"

    if any(w in text for w in ["backend", "back", "api", "server"]):
        return "backend"

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

# =========================
# RECOMMENDER
# =========================
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

    return [
        {
            "title": row["course_title"],
            "subject": row["subject"],
            "framework": row["FrameWork"],
            "language": row["Language"],
            "level": row["level"],
            "url": row["url"]
        }
        for _, row in results.iterrows()
    ]

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/chat")
def chat(query: str):
    if df is None:
        return {"error": "Dataset not loaded yet"}
    
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

# =========================
# RUN SERVER (LAST LINE ONLY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("🚀 Running on port:", port)
    uvicorn.run(app, host="0.0.0.0", port=port)