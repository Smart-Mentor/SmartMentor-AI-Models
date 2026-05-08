from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import random
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("DataSets/Data Model2.csv")

df.fillna("", inplace=True)
df.replace("-", "", inplace=True)

# =========================================================
# CREATE TEXT COLUMN
# =========================================================

df["text"] = (
    df["course_title"] + " " +
    df["subject"] + " " +
    df["FrameWork"] + " " +
    df["level"] + " " +
    df["Language"]
)

# =========================================================
# ML MODELS
# =========================================================

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["text"])

X = tfidf_matrix.toarray()

y = np.random.rand(len(df))

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X, y)

# =========================================================
# LIGHTWEIGHT BERT MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

# =========================================================
# VARIABLES
# =========================================================

subjects = df["subject"].str.lower().unique().tolist()

frameworks = df["FrameWork"].str.lower().unique().tolist()

languages = df["Language"].str.lower().unique().tolist()

levels = df["level"].str.lower().unique().tolist()

all_keywords = list(set(subjects + frameworks + languages))

# =========================================================
# WELCOME / EXIT
# =========================================================

welcome_statements = [
    "Welcome 👋 I'm your course recommendation assistant.",
    "Hello 🎓 Tell me what you want to learn.",
    "Hi 👋 What subject or framework are you interested in?",
]

greetings = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
]

exit_words = [
    "bye",
    "goodbye",
    "thanks",
    "thank you",
    "exit",
    "quit",
]

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="SmartMentor AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    message: str

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text):

    text = text.strip().lower()

    text = " ".join(text.split())

    return text


def correct_word(word):

    matches = difflib.get_close_matches(
        word,
        all_keywords,
        n=1,
        cutoff=0.85
    )

    return matches[0] if matches else None


def detect_subject(text):

    text = text.lower()

    for subject in subjects:

        if subject in text:
            return subject

    for word in text.split():

        suggestion = correct_word(word)

        if suggestion in subjects:
            return suggestion

    return None


def detect_framework(text):

    text = text.lower()

    for framework in frameworks:

        if framework.lower() in text:
            return framework.lower()

    for word in text.split():

        suggestion = correct_word(word)

        if suggestion in frameworks:
            return suggestion

    return None


def detect_language(text):

    text = text.lower()

    for language in languages:

        if language.lower() in text:
            return language.lower()

    return None


def detect_level(text):

    text = text.lower()

    beginner_words = [
        "beginner",
        "basic",
        "starter",
        "newbie",
    ]

    intermediate_words = [
        "intermediate",
        "medium",
        "mid",
    ]

    advanced_words = [
        "advanced",
        "expert",
        "professional",
    ]

    for word in beginner_words:

        if word in text:
            return "beginner level"

    for word in intermediate_words:

        if word in text:
            return "intermediate level"

    for word in advanced_words:

        if word in text:
            return "expert level"

    return None


def recommend_courses(
    subject=None,
    framework=None,
    language=None,
    level=None
):

    results = df.copy()

    if subject:

        results = results[
            results["subject"].str.lower() == subject
        ]

    if framework:

        results = results[
            results["FrameWork"].str.lower() == framework
        ]

    if language:

        results = results[
            results["Language"].str.lower() == language
        ]

    if level:

        results = results[
            results["level"].str.lower() == level
        ]

    if len(results) == 0:
        return []

    results = results.sample(frac=1)

    return results.head(4).to_dict(orient="records")


# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
async def root():

    return {
        "message": "SmartMentor AI API Running Successfully"
    }

# =========================================================
# CHAT ROUTE
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    user_input = clean_text(request.message)

    # Greetings

    if user_input in greetings:

        return {
            "message": random.choice(welcome_statements),
            "courses": []
        }

    # Exit

    if user_input in exit_words:

        return {
            "message": "Thanks for using SmartMentor 👋",
            "courses": []
        }

    # Detect Intent

    subject = detect_subject(user_input)

    framework = detect_framework(user_input)

    language = detect_language(user_input)

    level = detect_level(user_input)

    # Recommend Courses

    results = recommend_courses(
        subject,
        framework,
        language,
        level
    )

    # No Results

    if len(results) == 0:

        return {
            "message": "❌ No courses found.",
            "courses": []
        }

    # Format Courses

    courses = []

    for row in results:

        courses.append({
            "course_title": row["course_title"],
            "subject": row["subject"],
            "framework": row["FrameWork"],
            "language": row["Language"],
            "level": row["level"],
            "url": row["url"]
        })

    return {
        "message": "🎓 Courses Found Successfully",
        "courses": courses
    }