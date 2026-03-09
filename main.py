from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI Course Recommendation API")

# Load dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\archive\udemy_courses.csv")
df = df[['course_title', 'url', 'num_subscribers', 'subject', 'level', 'is_paid']]
df['level'] = df['level'].str.lower()

# Create combined text
df['combined_text'] = (
    df['course_title'] + " " +
    df['subject'] + " " +
    df['level']
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Request model
class UserRequest(BaseModel):
    message: str


# Detect payment
def detect_payment(text):
    text = text.lower()
    if "free" in text:
        return False
    if "paid" in text:
        return True
    return None


# Detect level
def detect_level(text):
    text = text.lower()
    if "beginner" in text:
        return "beginner level"
    if "intermediate" in text:
        return "intermediate level"
    if "advanced" in text or "expert" in text:
        return "expert level"
    return None


@app.post("/recommend")
def recommend_courses(request: UserRequest):

    user_input = request.message

    payment_filter = detect_payment(user_input)
    level_filter = detect_level(user_input)

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)

    similar_indices = similarity.argsort()[0][::-1]
    results = df.iloc[similar_indices]

    if level_filter:
        results = results[results['level'] == level_filter]

    if payment_filter is not None:
        results = results[results['is_paid'] == payment_filter]

    results = results.sort_values(by='num_subscribers', ascending=False)

    recommendations = []

    for _, row in results.head(5).iterrows():
        recommendations.append({
            "course_title": row['course_title'],
            "url": row['url'],
            "level": row['level'],
            "is_paid": row['is_paid'],
            "num_subscribers": int(row['num_subscribers'])
        })

    return {
        "message": "Top Course Recommendations",
        "results": recommendations
    }