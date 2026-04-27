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

# ========================== Original Data & Variables ==========================
df = pd.read_csv(r"D:\Games\¡\Data Model2.csv")
df.fillna("", inplace=True)
df.replace("-", "", inplace=True)

# Create combined text for ML models
df["text"] = (
    df["course_title"] + " " +
    df["subject"] + " " +
    df["FrameWork"] + " " +
    df["level"] + " " +
    df["Language"]
)

# Initialize ML models
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

X = tfidf_matrix.toarray()
y = np.random.rand(len(df))

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

subjects = df["subject"].str.lower().unique().tolist()
frameworks = df["FrameWork"].str.lower().unique().tolist()
languages = df["Language"].str.lower().unique().tolist()
levels = df["level"].str.lower().unique().tolist()
all_keywords = list(set(subjects + frameworks))

welcome_statements = [
    "Welcome! 👋 I'm your personal course recommendation assistant. What would you like to learn today?",
    "Hi there! Ready to discover the best courses for you? Just tell me a subject, framework, or level.",
    "Hello! 🎓 Let's find the perfect course together. What subject or skill are you interested in?",
    "Welcome to the Course Recommendation Chatbot! How can I help you level up your skills today?",
    "Hey! I'm here to recommend amazing courses. What topic, framework, or language are you looking for?"
]

greeting_responses = [
    "Hello! 👋 How can I help you today?",
    "Hi there! I'm here to help you find the best courses.",
    "Hey! What subject are you interested in?",
    "Hello! Looking for a course recommendation?",
    "Hi! Tell me what subject or framework you want to learn.",
    "Hey there! I can recommend courses for you.",
    "Hello! What would you like to learn today?",
    "Hi! Are you interested in Web Development, Backend Development, or something else?",
    "Hey! I can help you discover great courses.",
    "Hello! Just tell me the subject or framework you're looking for.",
    "Hi there! Ready to explore some courses?",
    "Hey! Let me know the subject and I'll suggest the best courses.",
    "Hello! I'm your course assistant. What do you want to learn?",
    "Hi! Feel free to ask about subjects or frameworks.",
    "Hey! Want beginner, intermediate, or advanced courses?"
]

exit_responses = [
    "You're welcome! 😊 If you need more course recommendations later, just come back.",
    "Bye! Happy learning! 📚 Feel free to return anytime.",
    "Glad I could help! See you next time — keep growing your skills!",
    "Thanks for chatting! Come back whenever you want more course suggestions.",
    "Take care! I'm always here if you need fresh recommendations. 👋"
]

exit_words = [
    "exit","quit","bye","goodbye","bye bye",
    "thanks","thanks!","thank you","thank you!",
    "thank you very much","thanks a lot","thanks alot","thx",
    "ok thanks","ok thank you",
    "no thanks","no thank you",
    "that's all","thats all",
    "done","finish","finished",
    "see you","see you later","good bye","stop"
]

greetings = [
    "hi","hello","hey","hey there","hi there","hello there",
    "good morning","good afternoon","good evening",
    "how are you","how are you doing",
    "how's it going","hows it going",
    "what's up","whats up","yo","sup","hiya",
    "greetings","nice to meet you","pleased to meet you",
    "long time no see","good to see you",
    "hello chatbot","hi chatbot","hey bot",
    "hello bot","hi assistant","hello assistant",
    "hey assistant","are you there","anyone there",
    "can you help me","i need help","help me",
    "start","let's start","lets start","begin",
    "let's begin","lets begin"
]

negation_words = ["not", "no", "don't", "dont", "isn't", "isnt", "aren't", "arent", "never", "without"]

beginner_words = [
    "beginner", "basic", "basics", "fundamental", "fundamentals",
    "intro", "introduction", "introductory", "starter", "start",
    "starting", "begin", "beginning", "newbie", "newcomer",
    "novice", "zero", "from scratch", "from zero", "no experience",
    "entry", "entry level", "entry-level", "elementary", "simple",
    "easy", "first", "first step", "first steps", "learn basics",
    "just starting", "just started", "new to", "brand new",
    "absolute beginner", "complete beginner", "total beginner",
    "fresh", "fresher", "groundup", "ground up"
]

intermediate_words = [
    "intermediate", "medium", "mid", "middle", "inter",
    "moderate", "average", "halfway", "in between", "inbetween",
    "some experience", "know basics", "familiar", "second level",
    "next level", "moving on", "already know", "improving",
    "level up", "level 2", "growing", "developing", "semi",
    "semi-advanced", "mediocre",
    "decent", "practiced", "somewhat experienced"
]

advanced_words = [
    "advanced", "expert", "pro", "professional", "professionals",
    "master", "mastery", "senior", "experienced", "veteran",
    "in depth", "in-depth", "hardcore",
    "high level", "high-level", "top level", "top-level",
    "skilled", "specialist", "guru", "ninja", "wizard",
    "proficient", "seasoned", "level 3", "hard", "complex",
    "challenging", "difficult", "tough", "comprehensive",
    "complete guide", "full course", "everything", "all topics",
    "advanced level", "expert level", "advanced course"
]

# ========================== Helper Functions ==========================
def normalize_spaces(text):
    return " ".join(text.split())

def clean_text(text):
    text = text.strip()
    text = normalize_spaces(text)
    text = text.lower()
    return text

def correct_word(word):
    matches = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.85)
    return matches[0] if matches else None

def is_input_related(text):
    text = text.lower()

    important_words = [
        "web", "frontend", "backend", "front", "back",
        "ai", "ml", "dl", "data", "mobile", "deep", "machine",
        "server", "api", "ui", "mob", "wp", "js", "desk",
        "ios", "ds", "dotnet", "boot", "bs", "artificial",
        "doc", "dock", "jq", "node", "node js", "py", "word",
    ]

    for word in text.split():
        if word in important_words:
            return True

        if word in subjects or word in frameworks or word in languages:
            return True

        suggestion = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.85)
        if suggestion:
            return True

    return False

def build_subject_aliases():
    aliases = {}
    for subject in subjects:
        words = subject.split("/")
        alias_list = []
        for w in words:
            alias_list.append(w)
            alias_list.append(w.replace(" ", ""))
        if "web" in subject or "frontend" in subject or "backend" in subject:
            alias_list += subject.replace("/", " ").split()
        aliases[subject] = list(set(alias_list))
    return aliases

subject_aliases = {
    "ai / artificial intelligence": [
        "ai",
        "artificial intelligence",
        "ai / artificial intelligence",
        "intelligent systems"
    ],
    "backend": [
        "backend",
        "back end",
        "back",
        "server",
        "server side",
        "api",
        "apis",
        "rest api",
    ],
    "web / frontend": [
        "front",
        "frontend",
        "front end",
        "web",
        "web development",
        "ui",
        "user interface",
        "website",
    ],
    "mobile": [
        "mob",
        "mobile",
        "android",
        "ios",
        "app development",
        "mobile app",
    ],
    "data science": [
        "data science",
        "ds",
        "data scientist",
        "data modeling"
    ],
    "data analysis": [
        "data analysis",
        "data analyst",
        "analysis",
        "analyzing data",
        "excel analysis"
    ],
    "cloud": [
        "cloud",
        "cloud computing",
        "azure",
        "gcp"
    ],
    "desktop": [
        "desk",
        "desktop",
        "desktop app",
        "windows app",
        "pc application"
    ],
    "java": [
        "java",
        "java development",
        "java programming"
    ]
}

framework_aliases = {
    "machinelearning": [
        "ml",
        "machine",
        "machine learning",
        "machine-learning",
        "machinelearn",
    ],
    "deeplearning": [
        "dl",
        "deep learning",
        "deep",
        "deep-learning",
        "deep learn",
        "neural networks",
    ],
    ".net": [
        "net",
        ".net",
        "dotnet",
        "asp.net",
        "asp net"
    ],
    "angular": [
        "angular",
        "angularjs",
    ],
    "aws": [
        "aws",
        "amazon web services",
        "aws cloud"
    ],
    "flutter": [
        "flutter",
        "dart flutter",
        "flutter framework"
    ],
    "bootstrap": [
        "boot",
        "bootstrap",
        "bootstrap framework",
        "bs"
    ],
    "docker": [
        "docker",
        "doc",
        "container",
        "containers",
        "docker container"
    ],
    "jquery": [
        "jquery",
        "jq"
    ],
    "native": [
        "native",
        "native development"
    ],
    "nodejs": [
        "node",
        "nodejs",
        "node js",
        "node.js"
    ],
    "php": [
        "php",
        "php language"
    ],
    "python": [
        "python",
        "py",
        "python language"
    ],
    "react": [
        "react",
        "reactjs",
        "react js",
        "react.js"
    ],
    "spring": [
        "spring",
        "spring boot",
        "springboot"
    ],
    "sql": [
        "sql",
        "database",
        "db",
        "structured query language"
    ],
    "wordpress": [
        "wordpress",
        "word press",
        "wp",
        "wordpress cms"
    ]
}

language_aliases = {
    "c#": [
        "csharp",
        "c sharp",
        "c#",
        "c #",
    ],
    "c++": [
        "cpp",
        "c p p",
        "c plus plus",
        "c++ language",
        "c+",
        "c++"
    ],
    "javascript": [
        "js",
        "java script",
        "javascript"
    ],
    "html": [
        "html",
        "html5",
        "hypertext markup",
        "hypertext markup language",
        "hyper text markup language",
    ],
    "css": [
        "css",
        "css3",
        "style sheet",
        "styling",
    ],
    "java": [
        "java",
        "java language",
        "java programming",
        "core java"
    ]
}

def detect_subject(text):
    text = text.lower()

    if any(word in text for word in ["web", "frontend", "front"]):
        return "web / frontend"

    if any(word in text for word in ["backend", "back", "api", "server"]):
        return "backend"

    if any(word in text for word in ["data science", "ds"]):
        return "data science"

    if any(word in text for word in ["analysis", "data analysis"]):
        return "data analysis"

    if any(word in text for word in ["ai", "artificial intelligence"]):
        return "ai / artificial intelligence"

    for subject, keywords in subject_aliases.items():
        for word in keywords:
            if word in text:
                return subject

    for s in subjects:
        if s in text:
            return s

    for word in text.split():
        suggestion = correct_word(word)
        if suggestion in subjects:
            return suggestion

    return None

def detect_framework(text):
    text = text.lower()
    text_no_space = text.replace(" ", "")

    for fw, aliases in framework_aliases.items():
        for alias in aliases:
            alias_lower = alias.lower()
            
            if alias_lower == "ml":
                if any(bad in text_no_space for bad in ["html", "htm"]):
                    continue
                if "ml" in text_no_space and not any(word in text for word in ["html", "html5", "html/css", "html css"]):
                    return fw
                continue
            
            if alias_lower in text or alias_lower in text_no_space:
                return fw

    for fw in frameworks:
        if fw.lower() in text or fw.lower().replace(" ", "") in text_no_space:
            return fw.lower()

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

def detect_level(text):
    text = text.lower()
    has_negation = any(neg in text for neg in negation_words)

    if has_negation:
        if any(word in text for word in beginner_words):
            return "not beginner level"
        elif any(word in text for word in intermediate_words):
            return "not intermediate level"
        elif any(word in text for word in advanced_words):
            return "not expert level"
        return None

    text_clean = text.replace(" and ", ",").replace(" or ", ",")
    parts = [p.strip() for p in text_clean.split(",") if p.strip()]
    detected = []
    
    for part in parts:
        found = False
        for word in beginner_words:
            if word in part:
                detected.append("beginner level")
                found = True
                break
        if not found:
            for word in intermediate_words:
                if word in part:
                    detected.append("intermediate level")
                    found = True
                    break
        if not found:
            for word in advanced_words:
                if word in part:
                    detected.append("expert level")
                    found = True
                    break
        if not found:
            for l in levels:
                if l in part:
                    detected.append(l)
                    found = True
                    break

    detected = list(dict.fromkeys(detected))

    if detected:
        return detected if len(detected) > 1 else detected[0]

    for word in beginner_words:
        if word in text:
            return "beginner level"
    for word in intermediate_words:
        if word in text:
            return "intermediate level"
    for word in advanced_words:
        if word in text:
            return "expert level"
    for l in levels:
        if l in text:
            return l
    return None

def show_options(subject):
    sub_df = df[df["subject"].str.lower() == subject]

    fw = sub_df["FrameWork"].unique()
    lang = sub_df["Language"].unique()

    has_framework = any(f.strip() for f in fw)
    has_language = any(l.strip() for l in lang)

    response = ""
    
    if has_framework:
        response += "\nAvailable Frameworks:\n"
        for f in fw:
            if f.strip() != "":
                response += f"- {f}\n"

    if has_language:
        response += "\nAvailable Languages:\n"
        for l in lang:
            if l.strip() != "":
                response += f"- {l}\n"

    if has_framework and has_language:
        response += "\n👉 Choose framework or language:"
    elif has_framework:
        response += "\n👉 Choose framework:"
    elif has_language:
        response += "\n👉 Choose language:"
    else:
        response += "\n❌ No frameworks or languages available for this subject."
    
    return response

def recommend_courses(subject=None, framework=None, level=None, language=None):
    results = df.copy()

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    if level is not None:
        if isinstance(level, list):
            level_lowers = [str(lvl).lower() for lvl in level]
            results = results[results["level"].str.lower().isin(level_lowers)]
        else:
            level_lower = str(level).lower()
            if level_lower.startswith("not "):
                exclude_level = level_lower.replace("not ", "").strip()
                results = results[results["level"].str.lower() != exclude_level]
            else:
                results = results[results["level"].str.lower() == level_lower]

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if len(results) == 0:
        return None

    results = results.sample(frac=1).reset_index(drop=True)
    return results.head(4)

def get_available_levels(subject=None, framework=None, language=None):
    results = df.copy()

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    levels_list = results["level"].unique()
    return [l for l in levels_list if l != ""]

def infer_subject(framework=None, language=None):
    if framework:
        result = df[df["FrameWork"].str.lower() == framework]
        if not result.empty:
            return result.iloc[0]["subject"].lower()

    if language:
        result = df[df["Language"].str.lower() == language]
        if not result.empty:
            return result.iloc[0]["subject"].lower()

    return None

def is_valid_combination(subject, framework=None, language=None):
    results = df.copy()

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    return not results.empty

def extract_intent(text):
    subject = detect_subject(text)
    framework = detect_framework(text)
    language = detect_language(text)
    level = detect_level(text)
    
    if not subject:
        subject = infer_subject(framework, language)

    return subject, framework, language, level

# Session management class to maintain conversation state
class ConversationSession:
    def __init__(self):
        self.state = {"subject": None, "framework": None, "language": None, "level": None}
        self.step = "start"
        self.waiting_for_response = False

    def reset(self):
        self.state = {"subject": None, "framework": None, "language": None, "level": None}
        self.step = "start"
        self.waiting_for_response = False

# Store active sessions (in production, use Redis or database)
sessions = {}

# ========================== FastAPI ==========================
app = FastAPI(title="SmartMentor AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    message: str
    courses: list = []
    session_id: str
    conversation_ended: bool = False

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = clean_text(request.message)
    
    # Get or create session
    session_id = request.session_id or str(random.randint(100000, 999999))
    if session_id not in sessions:
        sessions[session_id] = ConversationSession()
    
    session = sessions[session_id]
    
    # Check for greetings
    if any(g in user_input for g in greetings):
        return ChatResponse(
            message=random.choice(greeting_responses),
            courses=[],
            session_id=session_id,
            conversation_ended=False
        )
    
    # Check for exit
    if any(word in user_input for word in exit_words):
        session.reset()
        return ChatResponse(
            message=random.choice(exit_responses),
            courses=[],
            session_id=session_id,
            conversation_ended=True
        )
    
    # Main conversation flow
    if session.step == "start":
        if not is_input_related(user_input):
            return ChatResponse(
                message="❌ Please enter a valid course topic! (e.g. React, Python, Data Science, Backend...)",
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        smart_subject, smart_framework, smart_language, smart_level = extract_intent(user_input)
        
        if smart_level and (smart_framework or smart_language or smart_subject):
            # Build response message
            response_parts = []
            if isinstance(smart_level, list):
                response_parts.append(f"Levels: {', '.join([lvl.title() for lvl in smart_level])}")
            else:
                if smart_level.lower().startswith("not "):
                    response_parts.append(f"Level: Excluding {smart_level.replace('not ', '').strip()}")
                else:
                    response_parts.append(f"Level: {smart_level}")
            if smart_framework:
                response_parts.append(f"Framework: {smart_framework}")
            if smart_language:
                response_parts.append(f"Language: {smart_language}")
            if smart_subject:
                response_parts.append(f"Subject: {smart_subject}")
            
            session.state["subject"] = smart_subject
            session.state["framework"] = smart_framework
            session.state["language"] = smart_language
            session.state["level"] = smart_level
            
            results = recommend_courses(smart_subject, smart_framework, smart_level, smart_language)
            
            if results is None or len(results) == 0:
                return ChatResponse(
                    message="❌ No courses found for your request. Try changing the subject or level.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            
            courses = []
            for _, row in results.iterrows():
                courses.append({
                    "course_title": row["course_title"],
                    "level": row["level"],
                    "url": row["url"],
                    "framework": row.get("FrameWork", ""),
                    "language": row.get("Language", ""),
                    "subject": row.get("subject", "")
                })
            
            response_text = "\n".join(response_parts) + "\n\n🎓 Found Courses:\n\n"
            response_text += "\n\n".join([
                f"**{c['course_title']}**\n"
                f"Level: {c['level']}\n"
                f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                f"🔗 {c['url']}"
                for c in courses
            ])
            response_text += "\n\n👉 You can change the level for the same option or start a new search"
            
            session.step = "post_recommendation"
            session.waiting_for_response = True
            
            return ChatResponse(
                message=response_text,
                courses=courses,
                session_id=session_id,
                conversation_ended=False
            )
        
        subject = detect_subject(user_input)
        framework = detect_framework(user_input)
        language = detect_language(user_input)
        
        if framework:
            session.state["framework"] = framework
            session.state["subject"] = infer_subject(framework=framework)
            
            levels = get_available_levels(session.state["subject"], framework, None)
            
            if not levels:
                session.state["framework"] = None
                return ChatResponse(
                    message="❌ No courses found for this framework. Try another one.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            
            response_text = f"Subject: {session.state['subject']}\n"
            response_text += f"Detected framework: {framework}\n\n"
            response_text += "👉 Available Levels:\n" + "\n".join([f"- {l}" for l in levels])
            response_text += "\n\n👉 Choose level OR you can change subject:"
            
            session.step = "level"
            session.waiting_for_response = True
            
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        if language:
            session.state["language"] = language
            session.state["subject"] = infer_subject(language=language)
            
            levels = get_available_levels(session.state["subject"], None, language)
            
            response_text = "👉 Available Levels:\n" + "\n".join([f"- {l}" for l in levels])
            response_text += "\n\n👉 Choose level OR you can change language:"
            
            session.step = "level"
            session.waiting_for_response = True
            
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        if subject:
            session.state["subject"] = subject
            response_text = f"Subject: {subject}\n\n"
            response_text += show_options(subject)
            
            session.step = "framework_language"
            session.waiting_for_response = True
            
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        return ChatResponse(
            message="👉 Please enter a subject, framework, or language",
            courses=[],
            session_id=session_id,
            conversation_ended=False
        )
    
    elif session.step == "framework_language":
        if not is_input_related(user_input):
            return ChatResponse(
                message="❌ Please choose from the available options.",
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        framework = detect_framework(user_input)
        language = detect_language(user_input)
        
        if framework:
            if not is_valid_combination(session.state["subject"], framework=framework):
                return ChatResponse(
                    message="❌ This Framework does not belong to this subject.\n\n👉 Choose Language or Framework from the list.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            session.state["framework"] = framework
        
        if language:
            if not is_valid_combination(session.state["subject"], language=language):
                return ChatResponse(
                    message="❌ This Language does not belong to this subject.\n\n👉 Choose Language or Framework from the list.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            session.state["language"] = language
        
        if framework or language:
            levels = get_available_levels(
                session.state["subject"],
                session.state["framework"],
                session.state["language"]
            )
            
            if len(levels) == 1:
                session.state["level"] = levels[0]
                results = recommend_courses(
                    session.state["subject"],
                    session.state["framework"],
                    session.state["level"],
                    session.state["language"]
                )
                
                if results is None or len(results) == 0:
                    return ChatResponse(
                        message="❌ No courses found",
                        courses=[],
                        session_id=session_id,
                        conversation_ended=False
                    )
                
                courses = []
                for _, row in results.iterrows():
                    courses.append({
                        "course_title": row["course_title"],
                        "level": row["level"],
                        "url": row["url"],
                        "framework": row.get("FrameWork", ""),
                        "language": row.get("Language", ""),
                        "subject": row.get("subject", "")
                    })
                
                response_text = f"Only one level available: {levels[0]}\n\n🎓 Recommended Courses:\n\n"
                response_text += "\n\n".join([
                    f"**{c['course_title']}**\n"
                    f"Level: {c['level']}\n"
                    f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                    f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                    f"🔗 {c['url']}"
                    for c in courses
                ])
                response_text += "\n\n👉 Change the level or start a new search."
                
                session.step = "post_recommendation"
                session.waiting_for_response = True
                
                return ChatResponse(
                    message=response_text,
                    courses=courses,
                    session_id=session_id,
                    conversation_ended=False
                )
            
            elif len(levels) > 1:
                response_text = "👉 Available Levels:\n" + "\n".join([f"- {l}" for l in levels])
                response_text += "\n\n👉 Choose level:"
                session.step = "level"
                return ChatResponse(
                    message=response_text,
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
        
        return ChatResponse(
            message="❌ Invalid input!\n\nPlease enter a valid framework or language.",
            courses=[],
            session_id=session_id,
            conversation_ended=False
        )
    
    elif session.step == "level":
        if not is_input_related(user_input) and not detect_level(user_input):
            return ChatResponse(
                message="❌ Please enter a valid level.",
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        new_subject = detect_subject(user_input)
        new_framework = detect_framework(user_input)
        new_language = detect_language(user_input)
        
        if new_subject and new_subject != session.state["subject"]:
            session.state = {"subject": new_subject, "framework": None, "language": None, "level": None}
            response_text = show_options(new_subject)
            response_text += "\n\n👉 Choose framework or language:"
            session.step = "framework_language"
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        if new_framework and new_framework != session.state["framework"]:
            session.state["framework"] = new_framework
            session.state["language"] = None
            levels = get_available_levels(session.state["subject"], new_framework, None)
            
            if not levels:
                session.state["framework"] = None
                return ChatResponse(
                    message="❌ No courses found for this Framework.\n\n👉 Choose from available Frameworks in selected subject.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            
            response_text = "👉 Available Levels:\n" + "\n".join([f"- {l}" for l in levels])
            response_text += "\n\n👉 Choose level OR you can change framework:"
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        if new_language and new_language != session.state["language"]:
            session.state["language"] = new_language
            session.state["framework"] = None
            levels = get_available_levels(session.state["subject"], None, new_language)
            
            if not levels:
                session.state["language"] = None
                return ChatResponse(
                    message="❌ No courses found for this language.\n\n👉 Choose from available languages in selected subject.",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            
            response_text = "👉 Available Levels:\n" + "\n".join([f"- {l}" for l in levels])
            response_text += "\n\n👉 Choose level OR you can change language:"
            return ChatResponse(
                message=response_text,
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        level_detected = detect_level(user_input)
        levels = get_available_levels(
            session.state["subject"],
            session.state["framework"],
            session.state["language"]
        )
        available_lower = [l.lower() for l in levels]
        
        if level_detected:
            if isinstance(level_detected, list):
                valid_levels = [lvl for lvl in level_detected if str(lvl).lower() in available_lower]
                if valid_levels:
                    session.state["level"] = valid_levels
                    results = recommend_courses(
                        session.state["subject"],
                        session.state["framework"],
                        session.state["level"],
                        session.state["language"]
                    )
                    
                    if results is None or len(results) == 0:
                        return ChatResponse(
                            message="❌ No courses found",
                            courses=[],
                            session_id=session_id,
                            conversation_ended=False
                        )
                    
                    courses = []
                    for _, row in results.iterrows():
                        courses.append({
                            "course_title": row["course_title"],
                            "level": row["level"],
                            "url": row["url"],
                            "framework": row.get("FrameWork", ""),
                            "language": row.get("Language", ""),
                            "subject": row.get("subject", "")
                        })
                    
                    response_text = f"Levels: {', '.join([str(lvl).title() for lvl in valid_levels])}\n\n🎓 Courses:\n\n"
                    response_text += "\n\n".join([
                        f"**{c['course_title']}**\n"
                        f"Level: {c['level']}\n"
                        f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                        f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                        f"🔗 {c['url']}"
                        for c in courses
                    ])
                    response_text += "\n\n👉 Change the level or start a new search."
                    
                    session.step = "post_recommendation"
                    return ChatResponse(
                        message=response_text,
                        courses=courses,
                        session_id=session_id,
                        conversation_ended=False
                    )
                else:
                    return ChatResponse(
                        message="❌ None of the selected levels are available.",
                        courses=[],
                        session_id=session_id,
                        conversation_ended=False
                    )
            else:
                l_lower = level_detected.lower()
                if l_lower.startswith("not "):
                    excluded = l_lower.replace("not ", "").strip()
                    if excluded in available_lower:
                        session.state["level"] = level_detected
                        results = recommend_courses(
                            session.state["subject"],
                            session.state["framework"],
                            session.state["level"],
                            session.state["language"]
                        )
                        
                        if results is None or len(results) == 0:
                            return ChatResponse(
                                message=f"❌ No courses found after excluding {excluded.title()}",
                                courses=[],
                                session_id=session_id,
                                conversation_ended=False
                            )
                        
                        courses = []
                        for _, row in results.iterrows():
                            courses.append({
                                "course_title": row["course_title"],
                                "level": row["level"],
                                "url": row["url"],
                                "framework": row.get("FrameWork", ""),
                                "language": row.get("Language", ""),
                                "subject": row.get("subject", "")
                            })
                        
                        response_text = f"🎓 Courses (excluding {excluded.title()}):\n\n"
                        response_text += "\n\n".join([
                            f"**{c['course_title']}**\n"
                            f"Level: {c['level']}\n"
                            f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                            f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                            f"🔗 {c['url']}"
                            for c in courses
                        ])
                        response_text += "\n\n👉 Change the level or start a new search."
                        
                        session.step = "post_recommendation"
                        return ChatResponse(
                            message=response_text,
                            courses=courses,
                            session_id=session_id,
                            conversation_ended=False
                        )
                    else:
                        return ChatResponse(
                            message="❌ This level is not available to exclude.",
                            courses=[],
                            session_id=session_id,
                            conversation_ended=False
                        )
                else:
                    if l_lower in available_lower:
                        session.state["level"] = level_detected
                        results = recommend_courses(
                            session.state["subject"],
                            session.state["framework"],
                            session.state["level"],
                            session.state["language"]
                        )
                        
                        if results is None or len(results) == 0:
                            return ChatResponse(
                                message="❌ No courses found",
                                courses=[],
                                session_id=session_id,
                                conversation_ended=False
                            )
                        
                        courses = []
                        for _, row in results.iterrows():
                            courses.append({
                                "course_title": row["course_title"],
                                "level": row["level"],
                                "url": row["url"],
                                "framework": row.get("FrameWork", ""),
                                "language": row.get("Language", ""),
                                "subject": row.get("subject", "")
                            })
                        
                        response_text = "🎓 Courses:\n\n"
                        response_text += "\n\n".join([
                            f"**{c['course_title']}**\n"
                            f"Level: {c['level']}\n"
                            f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                            f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                            f"🔗 {c['url']}"
                            for c in courses
                        ])
                        response_text += "\n\n👉 Change the level or start a new search."
                        
                        session.step = "post_recommendation"
                        return ChatResponse(
                            message=response_text,
                            courses=courses,
                            session_id=session_id,
                            conversation_ended=False
                        )
        
        response_text = "Invalid level.\n\nAvailable:\n" + "\n".join([f"- {l}" for l in levels])
        return ChatResponse(
            message=response_text,
            courses=[],
            session_id=session_id,
            conversation_ended=False
        )
    
    elif session.step == "post_recommendation":
        text = clean_text(user_input)
        new_subject, new_framework, new_language, new_level = extract_intent(text)
        
        if new_level and not new_framework and not new_language:
            levels = get_available_levels(
                session.state["subject"],
                session.state["framework"],
                session.state["language"]
            )
            available_lower = [l.lower() for l in levels]
            
            if isinstance(new_level, list):
                valid_levels = [lvl for lvl in new_level if str(lvl).lower() in available_lower]
                if valid_levels:
                    session.state["level"] = valid_levels
                else:
                    return ChatResponse(
                        message="❌ None of these levels are available.",
                        courses=[],
                        session_id=session_id,
                        conversation_ended=False
                    )
            else:
                l_lower = new_level.lower()
                if l_lower.startswith("not "):
                    exclude_l = l_lower.replace("not ", "").strip()
                    if exclude_l in available_lower:
                        session.state["level"] = new_level
                    else:
                        return ChatResponse(
                            message="❌ This level is not available.",
                            courses=[],
                            session_id=session_id,
                            conversation_ended=False
                        )
                else:
                    if l_lower in available_lower:
                        session.state["level"] = new_level
                    else:
                        return ChatResponse(
                            message="❌ This level is not available.",
                            courses=[],
                            session_id=session_id,
                            conversation_ended=False
                        )
            
            results = recommend_courses(
                session.state["subject"],
                session.state["framework"],
                session.state["level"],
                session.state["language"]
            )
            
            if results is None or len(results) == 0:
                return ChatResponse(
                    message="❌ No courses found",
                    courses=[],
                    session_id=session_id,
                    conversation_ended=False
                )
            
            courses = []
            for _, row in results.iterrows():
                courses.append({
                    "course_title": row["course_title"],
                    "level": row["level"],
                    "url": row["url"],
                    "framework": row.get("FrameWork", ""),
                    "language": row.get("Language", ""),
                    "subject": row.get("subject", "")
                })
            
            response_text = "🎓 Updated Courses:\n\n"
            response_text += "\n\n".join([
                f"**{c['course_title']}**\n"
                f"Level: {c['level']}\n"
                f"{f'Framework: {c['framework']}\n' if c['framework'] else ''}"
                f"{f'Language: {c['language']}\n' if c['language'] else ''}"
                f"🔗 {c['url']}"
                for c in courses
            ])
            response_text += "\n\n👉 Change the level or start a new search."
            
            return ChatResponse(
                message=response_text,
                courses=courses,
                session_id=session_id,
                conversation_ended=False
            )
        
        if new_subject or new_framework or new_language:
            session.reset()
            return await chat(ChatRequest(message=request.message, session_id=session_id))
        
        if text in ["yes", "y"]:
            session.reset()
            return ChatResponse(
                message="👉 What subject, framework, or language do you want?",
                courses=[],
                session_id=session_id,
                conversation_ended=False
            )
        
        return ChatResponse(
            message="❌ I didn't understand!\n\nPlease enter a valid course or level.",
            courses=[],
            session_id=session_id,
            conversation_ended=False
        )
    
    return ChatResponse(
        message=random.choice(welcome_statements),
        courses=[],
        session_id=session_id,
        conversation_ended=False
    )

@app.get("/reset/{session_id}")
async def reset_session(session_id: str):
    if session_id in sessions:
        sessions[session_id].reset()
    return {"message": "Session reset successfully", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)