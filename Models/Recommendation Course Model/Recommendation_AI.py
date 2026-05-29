import pandas as pd
import numpy as np
import difflib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()

# =========================================================
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
    "Hey! Let me know the subject and I’ll suggest the best courses.",
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
    "end","see you","see you later","good bye","stop"
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

# Natural-language learning intent phrases — these strongly indicate the user
# is expressing a learning goal even without naming a subject keyword directly.
intent_phrases = [
    "i want to learn", "i wanna learn", "i'd like to learn", "i would like to learn",
    "i want to study", "i want courses", "i need courses", "i need a course",
    "i'm looking for courses", "i am looking for courses", "looking for courses",
    "recommend me courses", "suggest courses", "show me courses",
    "help me learn", "help me start", "how do i learn", "how to learn",
    "teach me", "i want to start", "i want to begin", "i want to get into",
    "how to get started", "getting started", "getting into",
    "i'm interested in", "i am interested in", "interested in learning",
    "i want to understand", "i want to know", "i want to know about",
    "fundamentals of", "basics of", "introduction to", "intro to",
    "beginner guide", "start learning", "start with", "learn the basics",
    "learn about", "learn how", "learn to build", "learn to create",
    "course for", "courses for", "course about", "courses about",
    "course on", "courses on",
]

# Subject-level goal descriptions that map to subjects even without keywords
subject_goal_phrases = {
    "backend": [
        "build apis", "build an api", "create a server", "server-side development",
        "how servers work", "rest apis", "create rest", "backend development",
        "server programming", "build web services", "web services",
    ],
    "web / frontend": [
        "build websites", "build a website", "create websites", "web pages",
        "design websites", "build web pages", "make a website", "front-end development",
        "ui development", "user interfaces", "web design",
    ],
    "mobile": [
        "build apps", "build an app", "mobile applications", "phone apps",
        "android development", "ios development", "create mobile apps",
    ],
    "data science": [
        "work with data", "analyze data", "machine learning models", "build ml models",
        "predict with data", "data pipelines",
    ],
    "ai / artificial intelligence": [
        "build ai", "artificial intelligence", "train models", "neural network",
        "deep learning models", "intelligent systems",
    ],
}

# =======================================================================

def normalize_spaces(text):
    return " ".join(text.split())   

def clean_text(text):
    text = text.strip()             
    text = normalize_spaces(text)   
    text = text.lower()             
    return text

df = pd.read_csv(r"D:\Games\AI MODELS\Models\Recommendation Course Model\DataSets\Data Model2.csv")

df.fillna("", inplace=True)
df.replace("-", "", inplace=True)

df["text"] = (
df["course_title"]+" "+
df["subject"]+" "+
df["FrameWork"]+" "+
df["level"]+" "+
df["Language"]
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

X = tfidf_matrix.toarray()
y = np.random.rand(len(df))

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X,y)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

subjects = df["subject"].str.lower().unique()
frameworks = df["FrameWork"].str.lower().unique()
levels = df["level"].str.lower().unique()
languages = df["Language"].str.lower().unique()

all_keywords = list(subjects) + list(frameworks)

def correct_word(word):
    matches = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.85)
    return matches[0] if matches else None

def is_input_related(text):
    text_lower = text.lower()

    # 1. Check if the message contains a learning-intent phrase
    for phrase in intent_phrases:
        if phrase in text_lower:
            return True

    # 2. Check subject-goal phrases (e.g. "build apis" → backend)
    for _, phrases in subject_goal_phrases.items():
        for phrase in phrases:
            if phrase in text_lower:
                return True

    important_words = [
        "web", "frontend", "backend", "front", "back",
        "ai", "ml", "dl", "data", "mobile", "deep", "machine",
        "server", "api", "ui", "mob", "wp", "js", "desk",
        "ios", "ds", "dotnet", "boot", "bs", "artificial",
        "doc", "dock", "jq", "node", "node js", "py", "word",
        "learn", "course", "courses", "study", "tutorial", "training",
        "programming", "development", "coding", "code", "software",
        "framework", "language", "skill", "skills", "technology",
    ]

    for word in text_lower.split():
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

# Smart clarifying questions per subject
subject_clarifying_questions = {
    "backend": (
        "Great choice! 🚀 For **Backend Development**, which framework or language are you interested in?\n"
        "For example: Node.js, Python, PHP, .NET, Spring (Java), or SQL.\n"
        "👉 Or just tell me if you have a preference — I'll guide you!"
    ),
    "web / frontend": (
        "Awesome! 🌐 For **Web/Frontend Development**, which framework or language are you thinking?\n"
        "For example: React, Angular, jQuery, Bootstrap, HTML, CSS, or JavaScript.\n"
        "👉 Let me know your preference!"
    ),
    "mobile": (
        "Nice! 📱 For **Mobile Development**, are you targeting Android, iOS, or cross-platform?\n"
        "For example: Flutter (cross-platform), React Native, or Android/iOS native.\n"
        "👉 Which one interests you?"
    ),
    "ai / artificial intelligence": (
        "Exciting! 🤖 For **AI/Machine Learning**, do you want to focus on Machine Learning, Deep Learning, or a specific tool like Python?\n"
        "👉 Tell me more about what you'd like to build!"
    ),
    "data science": (
        "Great! 📊 For **Data Science**, are you interested in a specific language like Python or SQL?\n"
        "👉 What kind of data work are you aiming for?"
    ),
    "data analysis": (
        "Good choice! 📈 For **Data Analysis**, do you want to work with SQL, Python, or Excel?\n"
        "👉 Let me know your preferred tool!"
    ),
    "cloud": (
        "Cloud is huge! ☁️ Are you interested in a specific platform like AWS, Azure, or GCP?\n"
        "👉 Which cloud provider do you want to learn?"
    ),
    "desktop": (
        "Solid pick! 💻 For **Desktop Development**, which language or framework do you prefer?\n"
        "For example: .NET (C#), Java, or Python.\n"
        "👉 Let me know!"
    ),
    "java": (
        "Java is powerful! ☕ Are you looking for core Java, Spring Boot, or something else?\n"
        "👉 What specifically do you want to build with Java?"
    ),
}

def get_clarifying_question(subject, available_frameworks, available_languages):
    """Return a smart clarifying question for the given subject, or fallback to listing options."""
    if subject in subject_clarifying_questions:
        question = subject_clarifying_questions[subject]
    else:
        question = f"For **{subject.title()}**, which framework or language are you interested in?"

    parts = [question]
    if available_frameworks:
        parts.append(f"\n📦 Available Frameworks: {', '.join(available_frameworks)}")
    if available_languages:
        parts.append(f"🗣️ Available Languages: {', '.join(available_languages)}")

    return "\n".join(parts)

subject_aliases = build_subject_aliases()

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

def detect_subject_from_goals(text):
    """Detect subject from natural-language goal descriptions."""
    text_lower = text.lower()
    for subject, phrases in subject_goal_phrases.items():
        for phrase in phrases:
            if phrase in text_lower:
                return subject
    return None

def strip_intent_phrases(text):
    """Remove leading intent phrases to expose the core topic."""
    text_lower = text.lower()
    for phrase in sorted(intent_phrases, key=len, reverse=True):
        if text_lower.startswith(phrase):
            text_lower = text_lower[len(phrase):].strip()
            break
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, " ").strip()
    return text_lower

def detect_subject(text):
    text_lower = text.lower()

    # First try on the full text
    if any(word in text_lower for word in ["web", "frontend", "front"]):
        if not any(word in text_lower for word in ["backend", "back", "server"]):
            return "web / frontend"

    if any(word in text_lower for word in ["backend", "back end", "server-side"]):
        return "backend"

    # Strip intent phrases and retry on the core topic
    core = strip_intent_phrases(text_lower)

    if any(word in core for word in ["web", "frontend", "front"]):
        return "web / frontend"

    if any(word in core for word in ["backend", "back", "api", "server", "back end"]):
        return "backend"

    if any(word in core for word in ["data science", "ds"]):
        return "data science"

    if any(word in core for word in ["analysis", "data analysis"]):
        return "data analysis"

    if any(word in core for word in ["ai", "artificial intelligence"]):
        return "ai / artificial intelligence"

    # Goal phrase detection (e.g. "build apis" → backend)
    goal_subject = detect_subject_from_goals(text_lower)
    if goal_subject:
        return goal_subject

    for subject, keywords in subject_aliases.items():
        for word in keywords:
            if word in core or word in text_lower:
                return subject

    for s in subjects:
        if s in core or s in text_lower:
            return s

    for word in core.split():
        suggestion = correct_word(word)
        if suggestion in subjects:
            return suggestion

    return None

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

def detect_framework(text):
    text = text.lower()
    text_no_space = text.replace(" ", "")
    avoid_ml = {"html", "html5", "html/css", "html css"}

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

def show_options(subject):
    sub_df = df[df["subject"].str.lower() == subject]

    fw = [f for f in sub_df["FrameWork"].unique() if f.strip()]
    lang = [l for l in sub_df["Language"].unique() if l.strip()]

    # Print the smart clarifying question instead of a bare list
    print("\n" + get_clarifying_question(subject, fw, lang))

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
    # Try detecting on both the original and the intent-stripped version
    subject = detect_subject(text)
    framework = detect_framework(text)
    language = detect_language(text)
    level = detect_level(text)

    # If no framework/language found on full text, try on core topic after stripping intent phrases
    if not framework and not language:
        core = strip_intent_phrases(text)
        if core != text.lower():
            framework = detect_framework(core)
            language = detect_language(core)

    if not subject:
        subject = infer_subject(framework, language)

    return subject, framework, language, level

def chatbot():
    print("\n========= Course Recommendation Chatbot =========\n")
    
    print("Chatbot:", random.choice(welcome_statements))
    
    state = {"subject": None, "framework": None, "language": None, "level": None}
    step = "start"

    user_input = None 

    while True:
        if user_input is None:
            user_input = clean_text(input("You: "))

        if user_input in greetings:
            print("\nChatbot:", random.choice(greeting_responses))
            user_input = None
            continue

        if user_input in exit_words:
            print("\nChatbot:", random.choice(exit_responses))
            break

        if step == "start":

            if not is_input_related(user_input):
                print("\n❌ Please enter a valid course topic!")
                user_input = None
                continue

            smart_subject, smart_framework, smart_language, smart_level = extract_intent(user_input)

            if smart_level and (smart_framework or smart_language or smart_subject):

                if isinstance(smart_level, list):
                    print("Levels:", ", ".join([lvl.title() for lvl in smart_level]))
                else:
                    if smart_level.lower().startswith("not "):
                        print("Level: Excluding", smart_level.replace("not ", "").strip())
                    else:
                        print("Level:", smart_level)
                if smart_framework:
                    print("Framework:", smart_framework)
                if smart_language:
                    print("Language:", smart_language)
                if smart_subject:
                    print("Subject:", smart_subject)

                state["subject"] = smart_subject
                state["framework"] = smart_framework
                state["language"] = smart_language
                state["level"] = smart_level

                results = recommend_courses(smart_subject, smart_framework, smart_level, smart_language)

                if results is None:
                    print("\n❌ No courses found")
                else:
                    print("\n🎓 Courses:\n")
                    for _, row in results.iterrows():
                        print("Title:", row["course_title"])
                        if row["FrameWork"] != "":
                            print("Framework:", row["FrameWork"])
                        if row["Language"] != "":
                            print("Language:", row["Language"])
                        print("Level:", row["level"])
                        print("URL:", row["url"])
                        print("-------------------")

                print("\n👉 You Can Change The Level For The Same Option Or Start a New Search\n")
                step = "post_recommendation"
                user_input = None
                continue

            subject = detect_subject(user_input)
            framework = detect_framework(user_input)
            language = detect_language(user_input)

            if framework:
                state["framework"] = framework
                state["subject"] = infer_subject(framework=framework)

                print(f"\nSubject: {state['subject']}")
                print(f"Detected framework: {framework}")

                levels = get_available_levels(state["subject"], framework, None)

                if not levels:
                    print("\n❌ No courses found for this framework.")
                    print("👉 Try another one.")

                    state["framework"] = None
                    step = "start"
                    user_input = None
                    continue

                print("\n👉 Available Levels:")
                for l in levels:
                    print("-", l)

                print("\n👉 Choose level OR You Can Change Subject:")
                step = "level"
                user_input = None
                continue

            if language:
                state["language"] = language
                state["subject"] = infer_subject(language=language)

                levels = get_available_levels(state["subject"], None, language)

                print("\n👉 Available Levels:")
                for l in levels:
                    print("-", l)

                print("\n👉 Choose level OR You Can Change Language:")
                step = "level"
                user_input = None
                continue

            if subject:
                state["subject"] = subject

                print(f"\nSubject: {subject.title()}")

                sub_df = df[df["subject"].str.lower() == subject]

                fw = [f for f in sub_df["FrameWork"].unique() if f.strip()]
                lang = [l for l in sub_df["Language"].unique() if l.strip()]

                has_framework = bool(fw)
                has_language = bool(lang)

                # If level was implied in the query (e.g. "fundamentals", "basics") store it now
                if smart_level and not state["level"]:
                    state["level"] = smart_level
                    if isinstance(smart_level, list):
                        print(f"Implied level: {', '.join([l.title() for l in smart_level])}")
                    else:
                        print(f"Implied level: {smart_level.title()}")

                if has_framework or has_language:
                    # Ask a smart, subject-specific clarifying question
                    print(get_clarifying_question(subject, fw, lang))
                    step = "framework_language"
                else:
                    print("\n❌ This subject has no data to continue.")
                    step = "start"

                user_input = None
                continue

            print("\n👉 Enter Subject, Framework or Language")
            user_input = None

        elif step == "framework_language":

            if not is_input_related(user_input):
                print("\n❌ Please choose from the available options.")
                user_input = None
                continue

            framework = detect_framework(user_input)
            language = detect_language(user_input)

            if framework:
                if not is_valid_combination(state["subject"], framework=framework):
                    print("\n❌ This Framework does not belong to this subject.")
                    print("\n👉 Choose Language or Framework from the list.")
                    user_input = None
                    continue
                
                state["framework"] = framework
                print(f"Framework: {framework}")

            if language:
                if not is_valid_combination(state["subject"], language=language):
                    print("\n❌ This Language does not belong to this subject.")
                    print("\n👉 Choose Language or Framework from the list.")
                    user_input = None
                    continue
                
                state["language"] = language
                print(f"Language: {language}")

            if framework or language:
                levels = get_available_levels(
                    state["subject"],
                    state["framework"],
                    state["language"]
                )

                if len(levels) == 1:
                    state["level"] = levels[0]
                    print(f"\nOnly one level available: {levels[0]}")
                    
                    results = recommend_courses(
                        state["subject"],
                        state["framework"],
                        state["level"],
                        state["language"]
                    )
                    
                    if results is None or len(results) == 0:
                        print("\n❌ No courses found")
                    else:
                        print("\n🎓 Recommended Courses:\n")
                        for _, row in results.iterrows():
                            print("Title:", row["course_title"])
                            if row["FrameWork"] != "":
                                print("Framework:", row["FrameWork"])
                            if row["Language"] != "":
                                print("Language:", row["Language"])
                            print("Level:", row["level"])
                            print("URL:", row["url"])
                            print("-------------------")
                    
                    print("\n👉 Change the level or Start a new search.\n")
                    step = "post_recommendation"
                    user_input = None
                    continue

                elif len(levels) > 1:
                    print("\n👉 Available Levels:")
                    for l in levels:
                        print("-", l)
                    print("\n👉 Choose level:")
                    step = "level"
                    user_input = None
                    continue

                else:
                    print("❌ No levels available for this selection.")
                    if framework:
                        state["framework"] = None
                    if language:
                        state["language"] = None
                    user_input = None
                    continue

            print("❌ Invalid input !")
            print("\n Please Enter A Valid Framework or Language .")
            user_input = None

        elif step == "level":

            if not is_input_related(user_input) and not detect_level(user_input):
                print("\n❌ Please enter a valid Level .")
                user_input = None
                continue

            new_subject = detect_subject(user_input)
            new_framework = detect_framework(user_input)
            new_language = detect_language(user_input)

            if new_subject and new_subject != state["subject"]:
                state = {"subject": new_subject, "framework": None, "language": None, "level": None}
                show_options(new_subject)
                print("\n👉 Choose framework or language:")
                step = "framework_language"
                user_input = None
                continue

            if new_framework and new_framework != state["framework"]:
                state["framework"] = new_framework
                state["language"] = None

                levels = get_available_levels(state["subject"], new_framework, None)

                if not levels:
                    print("\n❌ No courses found for this Framework.")
                    print("👉 Choose from available Frameworks in selected subject.\n")
                    state["framework"] = None
                    user_input = None
                    continue

                print("\n👉 Available Levels:")
                for l in levels:
                    print("-", l)

                print("\n👉 Choose level OR You Can Change Framework:")
                user_input = None
                continue

            if new_language and new_language != state["language"]:
                state["language"] = new_language
                state["framework"] = None

                levels = get_available_levels(state["subject"], None, new_language)

                if not levels:
                    print("\n❌ No courses found for this language.")
                    print("👉 Choose from available languages in selected subject.\n")
                    state["language"] = None
                    user_input = None
                    continue

                print("\n👉 Available Levels:")
                for l in levels:
                    print("-", l)

                print("\n👉 Choose level OR You Can Change Language:")
                user_input = None
                continue

            level_detected = detect_level(user_input)

            levels = get_available_levels(
                state["subject"],
                state["framework"],
                state["language"]
            )

            available_lower = [l.lower() for l in levels]

            if level_detected:
                if isinstance(level_detected, list):
                    valid_levels = [lvl for lvl in level_detected if str(lvl).lower() in available_lower]
                    if valid_levels:
                        state["level"] = valid_levels
                        print(f"\nLevels : {', '.join([str(lvl).title() for lvl in valid_levels])}")

                        results = recommend_courses(
                            state["subject"],
                            state["framework"],
                            state["level"],
                            state["language"]
                        )

                        if results is None:
                            print("\n❌ No courses found")
                        else:
                            print("\n🎓 Courses:\n")
                            for _, row in results.iterrows():
                                print("Title:", row["course_title"])
                                if row["FrameWork"] != "":
                                    print("Framework:", row["FrameWork"])
                                if row["Language"] != "":
                                    print("Language:", row["Language"])
                                print("Level:", row["level"])
                                print("URL:", row["url"])
                                print("-------------------")

                        print("\n👉 Change the level or Start a new search.\n")
                        step = "post_recommendation"
                        user_input = None
                        continue
                    else:
                        print("\n❌ None of the selected Levels are available.")
                        user_input = None
                        continue
                else:
                    l_lower = level_detected.lower()

                    if l_lower.startswith("not "):
                        excluded = l_lower.replace("not ", "").strip()
                        if excluded in available_lower:
                            print(f"\n Excluding level: {excluded.title()}")
                            remaining_levels = [l for l in levels if l.lower() != excluded]
                            if remaining_levels:
                                print("\n👉 Available Levels after exclusion:")
                                for l in remaining_levels:
                                    print("-", l)
                            state["level"] = level_detected
                            results = recommend_courses(
                                state["subject"],
                                state["framework"],
                                state["level"],
                                state["language"]
                            )
                            if results is None:
                                print("\n❌ No courses found after excluding this level")
                            else:
                                print(f"\n🎓 Courses (excluding {excluded.title()}):\n")
                                for _, row in results.iterrows():
                                    print("Title:", row["course_title"])
                                    if row["FrameWork"] != "":
                                        print("Framework:", row["FrameWork"])
                                    if row["Language"] != "":
                                        print("Language:", row["Language"])
                                    print("Level:", row["level"])
                                    print("URL:", row["url"])
                                    print("-------------------")
                            print("\n👉 Change the level or Start a new search.\n")
                            step = "post_recommendation"
                            user_input = None
                            continue
                        else:
                            print("\n❌ This level is not available to exclude.")
                            user_input = None
                            continue
                    else:
                        if l_lower in available_lower:
                            state["level"] = level_detected
                            results = recommend_courses(
                                state["subject"],
                                state["framework"],
                                state["level"],
                                state["language"]
                            )
                            if results is None:
                                print("\n❌ No courses found")
                            else:
                                print("\n🎓 Courses:\n")
                                for _, row in results.iterrows():
                                    print("Title:", row["course_title"])
                                    if row["FrameWork"] != "":
                                        print("Framework:", row["FrameWork"])
                                    if row["Language"] != "":
                                        print("Language:", row["Language"])
                                    print("Level:", row["level"])
                                    print("URL:", row["url"])
                                    print("-------------------")
                            print("\n👉 Change the level or Start a new search.\n")
                            step = "post_recommendation"
                            user_input = None
                            continue

            print("\nInvalid level.\n\nAvailable:")
            for l in levels:
                print("-", l)
            user_input = None

        elif step == "post_recommendation":

            text = clean_text(user_input)

            new_subject, new_framework, new_language, new_level = extract_intent(text)

            if new_level and not new_framework and not new_language:

                levels = get_available_levels(
                    state["subject"],
                    state["framework"],
                    state["language"]
                )

                available_lower = [l.lower() for l in levels]

                if isinstance(new_level, list):
                    valid_levels = [lvl for lvl in new_level if str(lvl).lower() in available_lower]
                    if valid_levels:
                        state["level"] = valid_levels
                        print(f"\nLevels updated to: {', '.join([str(lvl).title() for lvl in valid_levels])}")
                    else:
                        print("\n❌ None of these levels are available.")
                        user_input = None
                        continue
                else:
                    l_lower = new_level.lower()
                    if l_lower.startswith("not "):
                        exclude_l = l_lower.replace("not ", "").strip()
                        if exclude_l in available_lower:
                            state["level"] = new_level
                        else:
                            print("\n❌ This level is not available.")
                            user_input = None
                            continue
                    else:
                        if l_lower in available_lower:
                            state["level"] = new_level
                        else:
                            print("\n❌ This level is not available.")
                            user_input = None
                            continue

                results = recommend_courses(
                    state["subject"],
                    state["framework"],
                    state["level"],
                    state["language"]
                )

                if results is None:
                    print("\n❌ No courses found")
                else:
                    print("\n🎓 Updated Courses:\n")
                    for _, row in results.iterrows():
                        print("Title:", row["course_title"])
                        if row["FrameWork"] != "":
                            print("Framework:", row["FrameWork"])
                        if row["Language"] != "":
                            print("Language:", row["Language"])
                        print("Level:", row["level"])
                        print("URL:", row["url"])
                        print("-------------------")

                print("\n👉 Change the level or Start a new search.\n")
                user_input = None
                continue

            if new_subject or new_framework or new_language:
                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"
                user_input = text 
                continue

            if text in ["yes", "y"]:
                print("\n👉 What Subject, Framework or Language do you want?")
                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"
                user_input = None
                continue

            if text in ["no", "n"] or text in exit_words:
                print("\nChatbot:", random.choice(exit_responses))
                break

            print("\n❌ I didn't understand !")
            print("\nPlease Enter Valid Courses!\n")
            user_input = None
            continue

chatbot()