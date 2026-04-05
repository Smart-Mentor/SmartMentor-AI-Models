import pandas as pd
import numpy as np
import torch
import difflib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()

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

exit_words = [
"exit","quit","bye","goodbye","bye bye",
"thanks","thanks!","thank you","thank you!",
"thank you very much","thanks a lot","thx",
"ok thanks","ok thank you",
"no thanks","no thank you",
"that's all","thats all",
"done","finish","finished",
"end","see you","see you later","good bye","stop"
]

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

def detect_level(text):

    text = text.lower()

    if "beginner" in text:
        return "beginner level"

    if "intermediate" in text:
        return "intermediate level"

    if "advanced" in text or "expert" in text:
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

    if "web" in subject or "frontend" in subject:
        alias_list += subject.replace("/", " ").split()

    if "backend" in subject:
        alias_list += subject.replace("/", " ").split()

    subject_aliases[subject] = list(set(alias_list))  # remove duplicates

def recommend_courses(subject=None, framework=None, level=None, language=None):

    results = df.copy()

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    if level:
        results = results[results["level"].str.lower() == level]

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if len(results) == 0:
        return None

    return results.head(5)

##################################################
def get_available_levels(subject=None, framework=None, language=None):

    results = df.copy()

    if subject:
        results = results[results["subject"].str.lower() == subject]

    if framework:
        results = results[results["FrameWork"].str.lower() == framework]

    if language:
        results = results[results["Language"].str.lower() == language]

    levels = results["level"].unique()

    return [l for l in levels if l != ""]

#################################################
#################################################
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

def chatbot():

    print("\n========= Course Recommendation Chatbot =========\n")
    print("Hello! I'm here to help you find the best courses.\n")

    state = {
        "subject": None,
        "framework": None,
        "language": None,
        "level": None
    }

    step = "start"

    while True:

        raw_input = input("You: ")
        user_input = clean_text(raw_input)

        if user_input in greetings:
            print("\nChatbot:", random.choice(greeting_responses), "\n")
            continue

        if user_input in exit_words:
            print("\nChatbot: You're welcome! 👋 See you later.\n")
            break

        if step == "start":

            subject = detect_subject(user_input)
            framework = detect_framework(user_input)
            language = detect_language(user_input)
            level = detect_level(user_input)

            if framework:
                state["framework"] = framework
                state["subject"] = infer_subject(framework=framework)

                print(f"\nDetected framework: {framework}")
                print(f"Inferred subject: {state['subject']}")

                available_levels = get_available_levels(
                    state["subject"], framework, None
                )

                print("\n👉 Available Levels:")
                for l in available_levels:
                    print("-", l)

                print("\n👉 Please choose a level:")
                step = "level"
                continue

            if language:
                state["language"] = language
                state["subject"] = infer_subject(language=language)

                print(f"\nDetected language: {language}")
                print(f"Inferred subject: {state['subject']}")

                available_levels = get_available_levels(
                    state["subject"], None, language
                )

                print("\n👉 Available Levels:")
                for l in available_levels:
                    print("-", l)

                print("\n👉 Please choose a level:")
                step = "level"
                continue

            if subject:
                state["subject"] = subject

                print(f"\nSubject: {subject}")
                show_options(subject)

                print("\n👉 Choose a framework or language:")
                step = "framework_language"
                continue

            print("\n Please enter subject, framework, or language")

        elif step == "framework_language":

            framework = detect_framework(user_input)
            language = detect_language(user_input)

            if framework:
                state["framework"] = framework
                print(f"\nFramework selected: {framework}")

            if language:
                state["language"] = language
                print(f"\nLanguage selected: {language}")

            if framework or language:

                available_levels = get_available_levels(
                    state["subject"],
                    state["framework"],
                    state["language"]
                )

                print("\n👉 Available Levels:")
                for l in available_levels:
                    print("-", l)

                print("\n👉 Please choose a level:")
                step = "level"

            else:
                print("\n❌ Please choose a valid framework or language")

        elif step == "level":

            level = detect_level(user_input)

            available_levels = get_available_levels(
                state["subject"],
                state["framework"],
                state["language"]
            )

            if level and level in [l.lower() for l in available_levels]:

                state["level"] = level

                print(f"\nLevel selected: {level}")

                results = recommend_courses(
                    state["subject"],
                    state["framework"],
                    state["level"],
                    state["language"]
                )

                if results is None:
                    print("\n❌ No courses found\n")
                else:
                    print("\n🎓 Recommended Courses:\n")

                    for _, row in results.iterrows():

                        print("Course Title :", row["course_title"])

                        if row["FrameWork"]:
                            print("Framework    :", row["FrameWork"])

                        if row["Language"]:
                            print("Language     :", row["Language"])

                        print("Level        :", row["level"])
                        print("URL          :", row["url"])
                        print("--------------------------")

                print("\n👉 Do you want another recommendation? (yes / no)")
                step = "restart"

            else:
                print("\n❌ Invalid level. Available levels:\n")
                for l in available_levels:
                    print("-", l)


        elif step == "restart":

            if user_input in ["yes", "y"]:
                print("\n👉 What subject, framework, or language do you want?")
                step = "start"
                state = {"subject": None, "framework": None, "language": None, "level": None}
                continue

            if user_input.startswith("yes "):
                user_input = user_input.replace("yes ", "")

                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"

                continue

            if user_input in ["no", "n"] or user_input in exit_words:
                print("\nChatbot: Great! See you next time 👋")
                break

            else:
                state = {"subject": None, "framework": None, "language": None, "level": None}
                step = "start"
                continue
chatbot()
