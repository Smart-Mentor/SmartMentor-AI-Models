"""
Course Recommendation FastAPI — Hugging Face Spaces
Run:  uvicorn app:app --host 0.0.0.0 --port 7860
Docs: https://YOUR_SPACE.hf.space/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import difflib
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH = os.getenv("DATASET_PATH", "data_model2.csv")

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Course Recommendation API",
    description="AI-powered course recommendation chatbot exposed as a REST API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD DATA & MODELS (once at startup) ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.fillna("", inplace=True)
df.replace("-", "", inplace=True)

df["text"] = (
    df["course_title"] + " " +
    df["subject"]      + " " +
    df["FrameWork"]    + " " +
    df["level"]        + " " +
    df["Language"]
)

tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

X  = tfidf_matrix.toarray()
y  = np.random.rand(len(df))
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)

subjects   = df["subject"].str.lower().unique()
frameworks = df["FrameWork"].str.lower().unique()
levels_col = df["level"].str.lower().unique()
languages  = df["Language"].str.lower().unique()

all_keywords = list(subjects) + list(frameworks)

# ─── STATIC DATA ──────────────────────────────────────────────────────────────
welcome_statements = [
    "Welcome! 👋 I'm your personal course recommendation assistant. What would you like to learn today?",
    "Hi there! Ready to discover the best courses for you? Just tell me a subject, framework, or level.",
    "Hello! 🎓 Let's find the perfect course together. What subject or skill are you interested in?",
]

greeting_responses = [
    "Hello! 👋 How can I help you today?",
    "Hi there! I'm here to help you find the best courses.",
    "Hey! What subject are you interested in?",
]

exit_responses = [
    "You're welcome! 😊 Come back anytime.",
    "Bye! Happy learning! 📚",
    "Glad I could help! Keep growing your skills!",
]

exit_words = {
    "exit","quit","bye","goodbye","bye bye","thanks","thanks!","thank you",
    "thank you!","thank you very much","thanks a lot","thanks alot","thx",
    "ok thanks","ok thank you","no thanks","no thank you","that's all",
    "thats all","done","finish","finished","end","see you","see you later",
    "good bye","stop"
}

greetings = {
    "hi","hello","hey","hey there","hi there","hello there","good morning",
    "good afternoon","good evening","how are you","how are you doing",
    "how's it going","hows it going","what's up","whats up","yo","sup",
    "hiya","greetings","nice to meet you","hello chatbot","hi chatbot",
    "hey bot","hello bot","hi assistant","hello assistant","hey assistant",
    "are you there","anyone there","can you help me","i need help","help me",
    "start","let's start","lets start","begin","let's begin","lets begin"
}

negation_words = ["not","no","don't","dont","isn't","isnt","aren't","arent","never","without"]

subject_aliases = {
    "ai / artificial intelligence": ["ai","artificial intelligence","intelligent systems"],
    "backend": ["backend","back end","back","server","server side","api","apis","rest api"],
    "web / frontend": ["front","frontend","front end","web","web development","ui","user interface","website"],
    "mobile": ["mob","mobile","android","ios","app development","mobile app"],
    "data science": ["data science","ds","data scientist","data modeling"],
    "data analysis": ["data analysis","data analyst","analysis","analyzing data","excel analysis"],
    "cloud": ["cloud","cloud computing","azure","gcp"],
    "desktop": ["desk","desktop","desktop app","windows app","pc application"],
    "java": ["java","java development","java programming"]
}

framework_aliases = {
    "machinelearning": ["ml","machine","machine learning","machine-learning","machinelearn"],
    "deeplearning": ["dl","deep learning","deep","deep-learning","neural networks"],
    ".net": ["net",".net","dotnet","asp.net","asp net"],
    "angular": ["angular","angularjs"],
    "aws": ["aws","amazon web services","aws cloud"],
    "flutter": ["flutter","dart flutter","flutter framework"],
    "bootstrap": ["boot","bootstrap","bootstrap framework","bs"],
    "docker": ["docker","doc","container","containers","docker container"],
    "jquery": ["jquery","jq"],
    "native": ["native","native development"],
    "nodejs": ["node","nodejs","node js","node.js"],
    "php": ["php","php language"],
    "python": ["python","py","python language"],
    "react": ["react","reactjs","react js","react.js"],
    "spring": ["spring","spring boot","springboot"],
    "sql": ["sql","database","db","structured query language"],
    "wordpress": ["wordpress","word press","wp","wordpress cms"]
}

language_aliases = {
    "c#": ["csharp","c sharp","c#","c #"],
    "c++": ["cpp","c p p","c plus plus","c++ language","c+","c++"],
    "javascript": ["js","java script","javascript"],
    "html": ["html","html5","hypertext markup","hypertext markup language"],
    "css": ["css","css3","style sheet","styling"],
    "java": ["java","java language","java programming","core java"]
}

beginner_words     = ["beginner","basic","basics","fundamental","intro","introduction","starter","novice","zero","from scratch","entry level","easy","first step","brand new","absolute beginner","complete beginner"]
intermediate_words = ["intermediate","medium","mid","middle","moderate","average","familiar","level up","improving","level 2","semi"]
advanced_words     = ["advanced","expert","pro","professional","master","senior","experienced","in depth","hardcore","skilled","specialist","proficient","level 3","hard","complex","comprehensive","complete guide","full course"]

# ─── NLP HELPERS ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()

def correct_word(word):
    m = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.85)
    return m[0] if m else None

def detect_subject(text):
    t = text.lower()
    if any(w in t for w in ["web","frontend","front"]): return "web / frontend"
    if any(w in t for w in ["backend","back","api","server"]): return "backend"
    if any(w in t for w in ["data science","ds"]): return "data science"
    if any(w in t for w in ["analysis","data analysis"]): return "data analysis"
    if any(w in t for w in ["ai","artificial intelligence"]): return "ai / artificial intelligence"
    for subject, kws in subject_aliases.items():
        if any(kw in t for kw in kws): return subject
    for s in subjects:
        if s in t: return s
    for w in t.split():
        sg = correct_word(w)
        if sg in subjects: return sg
    return None

def detect_framework(text):
    t  = text.lower()
    tn = t.replace(" ", "")
    for fw, aliases in framework_aliases.items():
        for alias in aliases:
            al = alias.lower()
            if al == "ml":
                if "ml" in tn and not any(w in t for w in ["html","html5","html/css","html css"]): return fw
                continue
            if al in t or al in tn: return fw
    for fw in frameworks:
        if fw.lower() in t or fw.lower().replace(" ","") in tn: return fw.lower()
    return None

def detect_language(text):
    t = text.lower()
    for lang, aliases in language_aliases.items():
        if any(a in t for a in aliases): return lang
    for lang in languages:
        if lang in t: return lang
    return None

def detect_level(text):
    t = text.lower()
    has_neg = any(n in t for n in negation_words)
    if has_neg:
        if any(w in t for w in beginner_words):    return "not beginner level"
        if any(w in t for w in intermediate_words): return "not intermediate level"
        if any(w in t for w in advanced_words):    return "not expert level"
        return None
    parts    = [p.strip() for p in t.replace(" and ",",").replace(" or ",",").split(",") if p.strip()]
    detected = []
    for part in parts:
        found = False
        for w in beginner_words:
            if w in part: detected.append("beginner level"); found=True; break
        if not found:
            for w in intermediate_words:
                if w in part: detected.append("intermediate level"); found=True; break
        if not found:
            for w in advanced_words:
                if w in part: detected.append("expert level"); found=True; break
        if not found:
            for l in levels_col:
                if l in part: detected.append(l); found=True; break
    detected = list(dict.fromkeys(detected))
    if detected: return detected if len(detected) > 1 else detected[0]
    for w in beginner_words:
        if w in t: return "beginner level"
    for w in intermediate_words:
        if w in t: return "intermediate level"
    for w in advanced_words:
        if w in t: return "expert level"
    for l in levels_col:
        if l in t: return l
    return None

def infer_subject(framework=None, language=None):
    if framework:
        r = df[df["FrameWork"].str.lower() == framework]
        if not r.empty: return r.iloc[0]["subject"].lower()
    if language:
        r = df[df["Language"].str.lower() == language]
        if not r.empty: return r.iloc[0]["subject"].lower()
    return None

def is_input_related(text):
    important = {"web","frontend","backend","front","back","ai","ml","dl","data","mobile","deep",
                 "machine","server","api","ui","mob","wp","js","desk","ios","ds","dotnet","boot",
                 "bs","artificial","doc","dock","jq","node","py","word"}
    for word in text.split():
        if word in important: return True
        if word in subjects or word in frameworks or word in languages: return True
        if difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.85): return True
    return False

def extract_intent(text):
    subject   = detect_subject(text)
    framework = detect_framework(text)
    language  = detect_language(text)
    level     = detect_level(text)
    if not subject:
        subject = infer_subject(framework, language)
    return subject, framework, language, level

def recommend_courses(subject=None, framework=None, level=None, language=None):
    r = df.copy()
    if framework: r = r[r["FrameWork"].str.lower() == framework]
    if language:  r = r[r["Language"].str.lower() == language]
    if level is not None:
        if isinstance(level, list):
            lowers = [str(l).lower() for l in level]
            r = r[r["level"].str.lower().isin(lowers)]
        else:
            ll = str(level).lower()
            if ll.startswith("not "):
                r = r[r["level"].str.lower() != ll.replace("not ","").strip()]
            else:
                r = r[r["level"].str.lower() == ll]
    if subject: r = r[r["subject"].str.lower() == subject]
    if len(r) == 0: return None
    return r.sample(frac=1).reset_index(drop=True).head(4)

def get_available_levels(subject=None, framework=None, language=None):
    r = df.copy()
    if subject:   r = r[r["subject"].str.lower() == subject]
    if framework: r = r[r["FrameWork"].str.lower() == framework]
    if language:  r = r[r["Language"].str.lower() == language]
    return [l for l in r["level"].unique() if l != ""]

def courses_to_list(df_result):
    out = []
    for _, row in df_result.iterrows():
        c = {"title": row["course_title"], "level": row["level"], "url": row["url"]}
        if row["FrameWork"]: c["framework"] = row["FrameWork"]
        if row["Language"]:  c["language"]  = row["Language"]
        out.append(c)
    return out

# ─── SESSION STORE ────────────────────────────────────────────────────────────
sessions: dict = {}

def get_session(sid: str):
    if sid not in sessions:
        sessions[sid] = {
            "state": {"subject": None, "framework": None, "language": None, "level": None},
            "step": "start"
        }
    return sessions[sid]

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class RecommendRequest(BaseModel):
    subject:   Optional[str] = None
    framework: Optional[str] = None
    language:  Optional[str] = None
    level:     Optional[str] = None

class ResetRequest(BaseModel):
    session_id: str = "default"

class Course(BaseModel):
    title:     str
    level:     str
    url:       str
    framework: Optional[str] = None
    language:  Optional[str] = None

class ChatResponse(BaseModel):
    reply:            str
    courses:          List[Course] = []
    step:             str
    session_id:       str
    available_levels: List[str] = []

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "status": "running 🚀",
        "docs":   "/docs",
        "endpoints": ["/welcome", "/options", "/recommend", "/chat", "/reset"]
    }

@app.get("/welcome", tags=["General"])
def welcome():
    return {"message": random.choice(welcome_statements)}

@app.get("/options", tags=["General"])
def options():
    return {
        "subjects":   sorted([s for s in subjects if s]),
        "frameworks": sorted([f for f in frameworks if f]),
        "languages":  sorted([l for l in languages if l]),
        "levels":     sorted([l for l in levels_col if l])
    }

@app.post("/recommend", tags=["Recommendation"])
def recommend(req: RecommendRequest):
    results = recommend_courses(req.subject, req.framework, req.level, req.language)
    if results is None:
        return {"courses": [], "message": "❌ No courses found for your query."}
    return {"courses": courses_to_list(results)}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    sid        = req.session_id
    user_input = clean_text(req.message)

    if not user_input:
        return ChatResponse(reply="Please enter a message.", step="start", session_id=sid)

    session = get_session(sid)
    state   = session["state"]
    step    = session["step"]

    # ── Greetings ──
    if user_input in greetings:
        session["step"] = "start"
        return ChatResponse(reply=random.choice(greeting_responses), step="start", session_id=sid)

    # ── Exit ──
    if user_input in exit_words:
        sessions.pop(sid, None)
        return ChatResponse(reply=random.choice(exit_responses), step="exit", session_id=sid)

    # ══ STEP: start ══════════════════════════════════════════════════════════
    if step == "start":
        if not is_input_related(user_input):
            return ChatResponse(
                reply="❌ Please enter a valid course topic (subject, framework, or language).",
                step="start", session_id=sid
            )

        s_sub, s_fw, s_lang, s_lvl = extract_intent(user_input)

        # Full intent → recommend immediately
        if s_lvl and (s_fw or s_lang or s_sub):
            state.update(subject=s_sub, framework=s_fw, language=s_lang, level=s_lvl)
            results = recommend_courses(s_sub, s_fw, s_lvl, s_lang)
            session["step"] = "post_recommendation"
            session["state"] = state
            if results is None:
                return ChatResponse(reply="❌ No courses found.", step="post_recommendation", session_id=sid)
            return ChatResponse(
                reply="🎓 Here are your recommended courses:",
                courses=courses_to_list(results),
                step="post_recommendation", session_id=sid
            )

        # Framework only → ask level
        if s_fw:
            state["framework"] = s_fw
            state["subject"]   = infer_subject(framework=s_fw)
            lvls = get_available_levels(state["subject"], s_fw, None)
            if not lvls:
                state["framework"] = None
                return ChatResponse(reply=f"❌ No courses for '{s_fw}'. Try another framework.", step="start", session_id=sid)
            session["step"] = "level"
            session["state"] = state
            return ChatResponse(
                reply=f"Framework: **{s_fw}**\n👉 Available levels: {', '.join(lvls)}\nChoose a level:",
                step="level", session_id=sid, available_levels=lvls
            )

        # Language only → ask level
        if s_lang:
            state["language"] = s_lang
            state["subject"]  = infer_subject(language=s_lang)
            lvls = get_available_levels(state["subject"], None, s_lang)
            session["step"] = "level"
            session["state"] = state
            return ChatResponse(
                reply=f"Language: **{s_lang}**\n👉 Available levels: {', '.join(lvls)}\nChoose a level:",
                step="level", session_id=sid, available_levels=lvls
            )

        # Subject only → ask framework/language
        if s_sub:
            state["subject"] = s_sub
            sub_df    = df[df["subject"].str.lower() == s_sub]
            fw_list   = [f for f in sub_df["FrameWork"].unique() if f.strip()]
            lang_list = [l for l in sub_df["Language"].unique() if l.strip()]
            parts = [f"Subject: **{s_sub}**"]
            if fw_list:   parts.append(f"Frameworks: {', '.join(fw_list)}")
            if lang_list: parts.append(f"Languages: {', '.join(lang_list)}")
            parts.append("👉 Choose a framework or language:")
            session["step"] = "framework_or_language"
            session["state"] = state
            return ChatResponse(reply="\n".join(parts), step="framework_or_language", session_id=sid)

        return ChatResponse(reply="❌ Could not detect subject/framework/language. Try again.", step="start", session_id=sid)

    # ══ STEP: framework_or_language ══════════════════════════════════════════
    elif step == "framework_or_language":
        fw   = detect_framework(user_input)
        lang = detect_language(user_input)

        if fw:
            state["framework"] = fw
            lvls = get_available_levels(state["subject"], fw, state["language"])
            if not lvls:
                state["framework"] = None
                return ChatResponse(reply=f"❌ No courses for '{fw}'. Try another.", step=step, session_id=sid)
            session["step"] = "level"
            session["state"] = state
            return ChatResponse(
                reply=f"Framework: **{fw}**\n👉 Levels: {', '.join(lvls)}\nChoose a level:",
                step="level", session_id=sid, available_levels=lvls
            )
        if lang:
            state["language"] = lang
            lvls = get_available_levels(state["subject"], state["framework"], lang)
            session["step"] = "level"
            session["state"] = state
            return ChatResponse(
                reply=f"Language: **{lang}**\n👉 Levels: {', '.join(lvls)}\nChoose a level:",
                step="level", session_id=sid, available_levels=lvls
            )
        return ChatResponse(reply="❌ Please specify a framework or language.", step=step, session_id=sid)

    # ══ STEP: level ══════════════════════════════════════════════════════════
    elif step == "level":
        # User switching topic?
        ns, nfw, nlang, _ = extract_intent(user_input)
        if ns or nfw or nlang:
            session["state"] = {"subject": None, "framework": None, "language": None, "level": None}
            session["step"]  = "start"
            return ChatResponse(reply="👍 Starting a new search. What would you like to learn?", step="start", session_id=sid)

        level_det   = detect_level(user_input)
        avail       = get_available_levels(state["subject"], state["framework"], state["language"])
        avail_lower = [l.lower() for l in avail]

        if not level_det:
            return ChatResponse(reply=f"❌ Level not recognised. Available: {', '.join(avail)}", step="level", session_id=sid, available_levels=avail)

        if isinstance(level_det, list):
            valid = [l for l in level_det if str(l).lower() in avail_lower]
            if not valid:
                return ChatResponse(reply="❌ None of the selected levels are available.", step="level", session_id=sid, available_levels=avail)
            state["level"] = valid
        else:
            ll = level_det.lower()
            if ll.startswith("not "):
                if ll.replace("not ","").strip() not in avail_lower:
                    return ChatResponse(reply="❌ That level is not available to exclude.", step="level", session_id=sid, available_levels=avail)
            elif ll not in avail_lower:
                return ChatResponse(reply=f"❌ Level not available. Choose from: {', '.join(avail)}", step="level", session_id=sid, available_levels=avail)
            state["level"] = level_det

        results = recommend_courses(state["subject"], state["framework"], state["level"], state["language"])
        session["step"] = "post_recommendation"
        session["state"] = state
        if results is None:
            return ChatResponse(reply="❌ No courses found.", step="post_recommendation", session_id=sid)
        return ChatResponse(
            reply="🎓 Here are your recommended courses:",
            courses=courses_to_list(results),
            step="post_recommendation", session_id=sid
        )

    # ══ STEP: post_recommendation ═════════════════════════════════════════════
    elif step == "post_recommendation":
        ns, nfw, nlang, nlvl = extract_intent(user_input)

        # Level change only
        if nlvl and not nfw and not nlang and not ns:
            avail       = get_available_levels(state["subject"], state["framework"], state["language"])
            avail_lower = [l.lower() for l in avail]
            if isinstance(nlvl, list):
                valid = [l for l in nlvl if str(l).lower() in avail_lower]
                if not valid:
                    return ChatResponse(reply="❌ None of these levels are available.", step=step, session_id=sid, available_levels=avail)
                state["level"] = valid
            else:
                ll = nlvl.lower()
                if ll.startswith("not "):
                    if ll.replace("not ","").strip() not in avail_lower:
                        return ChatResponse(reply="❌ That level is not available.", step=step, session_id=sid, available_levels=avail)
                elif ll not in avail_lower:
                    return ChatResponse(reply="❌ That level is not available.", step=step, session_id=sid, available_levels=avail)
                state["level"] = nlvl
            results = recommend_courses(state["subject"], state["framework"], state["level"], state["language"])
            session["state"] = state
            if results is None:
                return ChatResponse(reply="❌ No courses found.", step=step, session_id=sid)
            return ChatResponse(reply="🎓 Updated courses:", courses=courses_to_list(results), step=step, session_id=sid)

        # New search
        if ns or nfw or nlang:
            session["state"] = {"subject": None, "framework": None, "language": None, "level": None}
            session["step"]  = "start"
            return ChatResponse(reply="👍 New search! What would you like to learn?", step="start", session_id=sid)

        if user_input in ["yes","y"]:
            session["state"] = {"subject": None, "framework": None, "language": None, "level": None}
            session["step"]  = "start"
            return ChatResponse(reply="👉 What subject, framework, or language are you looking for?", step="start", session_id=sid)

        if user_input in ["no","n"] or user_input in exit_words:
            sessions.pop(sid, None)
            return ChatResponse(reply=random.choice(exit_responses), step="exit", session_id=sid)

        return ChatResponse(reply="❌ I didn't understand. Change level, start a new search, or say 'done'.", step=step, session_id=sid)

    return ChatResponse(reply="Something went wrong. Please /reset.", step=step, session_id=sid)


@app.post("/reset", tags=["General"])
def reset(req: ResetRequest):
    sessions.pop(req.session_id, None)
    return {"message": f"Session '{req.session_id}' reset.", "welcome": random.choice(welcome_statements)}
