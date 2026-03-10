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

# GREETINGS
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
"Hi! Are you interested in Machine Learning, Deep Learning, or something else?",
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
"end","stop"
]

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\archive\newData.csv")
df.fillna("", inplace=True)

df["text"] = (
df["course_title"]+" "+
df["subject"]+" "+
df["FrameWork"]+" "+
df["level"]
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

all_keywords = list(subjects)+list(frameworks)

def correct_word(word):

    match = difflib.get_close_matches(word,all_keywords,n=1,cutoff=0.7)

    if match:
        print("Did you mean:",match[0],"?")
        return match[0]

    return None

def detect_subject(text):

    for word in text.split():

        if word in subjects:
            return word

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

    text=text.lower()

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

def show_options(subject):

    fw = df[df["subject"].str.lower()==subject]["FrameWork"].unique()
    lv = df[df["subject"].str.lower()==subject]["level"].unique()

    print("\nAvailable Frameworks:")

    for f in fw:
        print("-",f)

    print("\nAvailable Levels:")

    for l in lv:
        print("-",l)

def recommend_courses(subject=None,framework=None,level=None):

    results=df.copy()

    if subject:
        results=results[results["subject"].str.lower()==subject]

    if framework:
        results=results[results["FrameWork"].str.lower()==framework]

    if level:
        results=results[results["level"].str.lower()==level]

    if len(results)==0:
        return None

    return results.head(5)

def chatbot():

    print("\nAI Course Recommendation Chatbot\n")

    while True:

        user_input=input("You: ").lower()

        if user_input in greetings:

            print("\nChatbot:",random.choice(greeting_responses),"\n")
            continue

        if user_input in exit_words:

            print("\nChatbot: You're welcome! 😊 If you need more course recommendations later, come back anytime.\n")
            break

        subject=detect_subject(user_input)
        framework=detect_framework(user_input)
        level=detect_level(user_input)

        if subject and not framework:

            print("\nDetected subject:",subject)
            show_options(subject)
            print("\nNow type framework and level to get courses\n")
            continue

        if subject or framework:

            print("\nDetected:")

            if subject:
                print("Subject:",subject)

            if framework:
                print("Framework:",framework)

            if level:
                print("Level:",level)

            results=recommend_courses(subject,framework,level)

            if results is None:

                print("\nNo courses found\n")
                continue

            print("\nRecommended Courses:\n")

            for _,row in results.iterrows():

                print("Course Title :",row["course_title"])
                print("Framework    :",row["FrameWork"])
                print("Level        :",row["level"])
                print("URL          :",row["url"])
                print("--------------------------")

        else:

            print("\nI couldn't understand. Try mentioning a subject or framework.\n")


chatbot()