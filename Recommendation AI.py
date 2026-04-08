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

# =======================================================================

def normalize_spaces(text):
    return " ".join(text.split())   

def clean_text(text):
    text = text.strip()             
    text = normalize_spaces(text)   
    text = text.lower()             
    return text

df = pd.read_csv(r"D:\Games\¡\TryNewOneData.csv")

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
    match = difflib.get_close_matches(word, all_keywords, n=1, cutoff=0.7)
    if match:
        return match[0]
    return None

def detect_subject(text):
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
    for word in text.split():
        if word in frameworks:
            return word
        suggestion = correct_word(word)
        if suggestion in frameworks:
            return suggestion
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
    "deep", "deep dive", "in depth", "in-depth", "hardcore",
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

def detect_language(text):
    for word in text.split():
        if word in languages:
            return word
    return None

def show_options(subject):
    fw = df[df["subject"].str.lower() == subject]["FrameWork"].unique()
    lv = df[df["subject"].str.lower() == subject]["level"].unique()
    lang = df[df["subject"].str.lower() == subject]["Language"].unique()

    print("\nAvailable Frameworks:")
    for f in fw:
        if f != "":
            print("-", f)

    print("\nAvailable Languages:")
    for l in lang:
        if l != "":
            print("-", l)

subject_aliases = {}

for subject in df["subject"].str.lower().unique():
    words = subject.split("/")  
    alias_list = []
    for w in words:
        alias_list.append(w)
        alias_list.append(w.replace(" ", ""))  
    if "web" in subject or "frontend" in subject or "backend" in subject:
        alias_list += subject.replace("/", " ").split()
    subject_aliases[subject] = list(set(alias_list))  

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

            smart_subject, smart_framework, smart_language, smart_level = extract_intent(user_input)

            if smart_level and (smart_framework or smart_language or smart_subject):

                print("\n✅ Detected:")
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

                print("\n👉 Do you want to change the level for the same option? Or start a new search?")
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

                print("\n👉 Choose level OR You Can Change Framework/Language/Subject:")
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

                print(f"\nSubject: {subject}")
                show_options(subject)

                print("\n👉 Choose framework or language:")
                step = "framework_language"
                user_input = None
                continue

            print("\n❌ Enter subject, framework, or language")
            user_input = None

        elif step == "framework_language":

            framework = detect_framework(user_input)
            language = detect_language(user_input)

            if framework:
                if not is_valid_combination(state["subject"], framework=framework):
                    print("\n❌ This framework does not belong to this subject.")
                    print("\n👉 Choose language or framework from the list.")
                    user_input = None
                    continue
                
                state["framework"] = framework
                print(f"Framework: {framework}")

            if language:
                if not is_valid_combination(state["subject"], language=language):
                    print("\n❌ This language does not belong to this subject.")
                    print("\n👉 Choose language or framework from the list.")
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

                if not levels:
                    print("👉 Try another framework or language.")

                    state["framework"] = None
                    state["language"] = None

                    step = "framework_language"
                    user_input = None
                    continue

                print("\n👉 Available Levels:")
                for l in levels:
                    print("-", l)

                print("\n👉 Choose level OR You Can Change Framework/Language/Subject:")
                step = "level"
                user_input = None
                continue

            print("❌ Invalid input")
            user_input = None

        elif step == "level":

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
                    print("\n❌ No courses found for this framework.")
                    print("👉 Choose from available frameworks in selected subject.\n")
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
                        print(f"\n✅ Levels selected: {', '.join([str(lvl).title() for lvl in valid_levels])}")

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

                        print("\n👉 Do you want to change the level again or start a new search?")
                        step = "post_recommendation"
                        user_input = None
                        continue
                    else:
                        print("\n❌ None of the selected levels are available.")
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
                            print("\n👉 Do you want to change level again or start new search?")
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
                            print("\n👉 Do you want to change the level again or start a new search?")
                            step = "post_recommendation"
                            user_input = None
                            continue

            print("\n❌ Invalid level. Available:")
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

                print("\n👉 Do you want to change the level again or start a new search?")
                user_input = None
                continue

            if new_subject or new_framework or new_language:
                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"
                user_input = text 
                continue

            if text in ["yes", "y"]:
                print("\n👉 What subject, framework, or language do you want?")
                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"
                user_input = None
                continue

            if text in ["no", "n"] or text in exit_words:
                print("\nChatbot:", random.choice(exit_responses))
                break

            print("\n❓ I didn't understand.")
            user_input = None
            continue

chatbot()