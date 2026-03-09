import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()


df = pd.read_csv(r"C:\Users\Lenovo\Downloads\archive\newData.csv")

df.fillna("", inplace=True)

df["text"] = (
    df["course_title"]
    + " "
    + df["subject"]
    + " "
    + df["FrameWork"]
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

# RANDOM FOREST

X = tfidf_matrix.toarray()
y = np.random.rand(len(df))
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X, y)

# TRANSFORMER MODEL

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

subjects = df["subject"].unique()


def detect_subject(user_text):

    for s in subjects:
        if s.lower() in user_text.lower():
            return s
    return None


def detect_level(text):
    text = text.lower()

    if "beginner" in text:
        return "Beginner Level"
    
    if "intermediate" in text:
        return "Intermediate Level"
    
    if "expert" in text or "advanced" in text:
        return "Expert Level"

    return None


def show_frameworks(subject):

    frameworks = df[df["subject"] == subject]["FrameWork"].unique()
    print("\nAvailable Frameworks:\n")

    for f in frameworks:
        print("-", f)


def recommend_courses(user_input):
    user_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df["score"] = similarity
    level = detect_level(user_input)
    results = df.copy()

    if level:
        results = results[results["level"] == level]

    results = results.sort_values(by="score", ascending=False)
    return results.head(5)

def chatbot():

    print("\nAI Course Recommendation Chatbot \n")
    print("Hello! How can I help you?\n")
    selected_subject = None

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit" or user_input.lower() == "quit" or user_input.lower() == "bye" or user_input.lower() == "thanks" or user_input.lower() == "thank you" or user_input.lower() == "thank you!" or user_input.lower() == "thanks!" or user_input.lower() == "thank you very much" or user_input.lower() == "thanks a lot":

            print("Chatbot: Goodbye!")
            break

        subject = detect_subject(user_input)

        if subject:
            selected_subject = subject
            show_frameworks(subject)
            continue

        if selected_subject:
            query = selected_subject + " " + user_input

        else:
            query = user_input

        results = recommend_courses(query)
        print("\nTop Recommended Courses:\n")

        for _, row in results.iterrows():
            print("\nCourse Title :", row["course_title"])
            print("Framework    :", row["FrameWork"])
            print("Level        :", row["level"])
            print("URL          :", row["url"])
            print("----------------------------------")

chatbot()