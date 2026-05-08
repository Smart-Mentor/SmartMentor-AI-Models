import streamlit as st
import pandas as pd
import pickle
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="CV Matcher AI", layout="wide")

# ================================
# LOAD MODEL
# ================================
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("job_vectors.pkl", "rb") as f:
    job_vectors = pickle.load(f)

df = pd.read_pickle("data.pkl")

# ================================
# CLEAN TEXT
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = " ".join(text.split())
    return text

# ================================
# RECOMMEND FUNCTION
# ================================
def recommend_jobs(cv_text, top_n=3):
    cv_text = clean_text(cv_text)

    cv_vector = vectorizer.transform([cv_text])
    similarities = cosine_similarity(cv_vector, job_vectors)

    top_indices = similarities[0].argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][[
        "Job Title",
        "Company Name"
    ]].copy()

    results["score"] = similarities[0][top_indices]

    return results

# ================================
# UI DESIGN
# ================================
st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>AI CV Job Matcher</h1>
    <p style='text-align:center; color:gray;'>Upload your CV and get the best job recommendations instantly</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# UPLOAD FILE
# ================================
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

cv_text = ""

if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            cv_text += page_text

    st.success("CV uploaded successfully")

# ================================
# BUTTON
# ================================
if st.button("Find Best Jobs"):

    if cv_text == "":
        st.warning("Please upload a valid CV first")

    else:
        results = recommend_jobs(cv_text)

        st.markdown("## Top Job Matches")

        # ================================
        # CARDS UI
        # ================================
        for i, row in results.iterrows():
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 20px;
                    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
                    color: white;
                ">
                    <h3 style="color:#4CAF50;">{row['Job Title']}</h3>
                    <h4>{row['Company Name']}</h4>
                </div>
            """, unsafe_allow_html=True)

