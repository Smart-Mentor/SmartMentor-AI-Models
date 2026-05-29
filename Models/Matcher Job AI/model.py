"""
model.py — Core recommendation engine (framework-agnostic)
Called by both app.py (Gradio UI) and api.py (FastAPI REST)
"""

import re
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# CAREER SKILLS  (data-driven from dataset analysis)
# ─────────────────────────────────────────────────────────────

CAREER_SKILLS = {

    "Artificial Intelligence": [
        # core — confirmed in dataset
        "python", "python programming", "machine learning", "deep learning",
        "pytorch", "tensorflow", "artificial intelligence", "data preprocessing",
        "algorithm development", "data analysis",
        # NLP / CV — present in requirements
        "natural language processing", "natural language processing (nlp)",
        "nlp", "computer vision", "neural networks",
        # modern AI
        "large language models (llms)", "llm", "generative ai", "transformers",
        "huggingface", "reinforcement learning", "ai automation", "ai deployment",
        # MLOps / infra
        "mlops", "model deployment", "scikit-learn", "feature engineering",
        "containerization (docker/kubernetes)", "ci/cd automation",
        "cloud computing (aws, azure, gcp)", "cloud computing",
        "database management (sql/nosql)", "docker", "aws", "azure",
        # soft
        "problem solving", "software engineering",
    ],

    "Data Analysis": [
        # tools — confirmed high frequency
        "sql", "python", "excel", "microsoft excel", "power bi", "tableau",
        "looker", "mysql", "postgresql", "oracle", "sql server", "ms sql",
        # skills — confirmed
        "data analysis", "data analytics", "business analysis", "business analyst",
        "statistical analysis", "data visualization", "reporting", "etl",
        "data modeling", "data mining", "kpis", "dashboard", "pivot tables",
        "google analytics", "requirements gathering", "documentation",
        "business process analysis", "system integration", "erp implementation",
        # domain
        "business intelligence", "erp", "sap", "odoo", "crm",
        "product management", "analysis", "analyst",
        # soft
        "problem solving", "problem-solving", "team leadership",
    ],

    "Data Science": [
        # confirmed in dataset
        "machine learning", "python", "data science", "statistical analysis",
        "data mining", "data modeling", "hadoop", "big data",
        "big data technolog", "cloud technology",
        # tools
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
        "sql", "deep learning", "pytorch", "tensorflow",
        "hypothesis testing", "a/b testing", "regression", "classification",
        "clustering", "feature engineering", "model deployment",
    ],

    "Backend": [
        # languages — confirmed high frequency
        "python", "java", "php", "javascript", "c#", ".net", "asp.net",
        ".net core", "node.js", "node",
        # frameworks — confirmed
        "django", "flask", "laravel", "spring", "asp.net core",
        # databases — confirmed
        "sql", "postgresql", "mysql", "mongodb", "oracle", "sql server",
        "redis", "elasticsearch", "database management",
        # APIs / architecture — confirmed
        "backend development", "restful apis", "rest api", "graphql",
        "api integration", "microservices", "web services",
        # devops — confirmed
        "docker", "aws", "azure", "linux", "git", "ci/cd",
        "cloud computing", "networking", "ccna",
        # ERP — very high in dataset
        "odoo", "odoo development", "sap", "erp",
        "erp implementation", "erp system",
        # support/infra — confirmed
        "technical support", "technical troubleshooting", "troubleshooting",
        "network administration", "system administration",
        "cybersecurity", "network security",
        # soft — confirmed
        "software engineering", "team leadership", "project management",
    ],

    "Frontend": [
        # core — confirmed high frequency
        "html", "html5", "css", "css3", "javascript",
        "javascript (es6+)", "typescript",
        # frameworks — confirmed
        "react", "react.js", "angular", "vue", "vue 3",
        "next.js", "next14", "bootstrap", "tailwind", "tailwind css",
        "redux", "inertia.js",
        # tooling — confirmed
        "git", "restful apis", "restful api", "version control (git)",
        "webpack", "front-end development", "front-end", "web development",
        # design collab — confirmed
        "ui/ux", "ui/ux collaboration", "ui/ux collaboratio",
        "responsive design", "figma", "adobe xd",
        # soft
        "agile", "agile methodologies", "software engineering",
    ],

    "Full Stack": [
        # frontend — confirmed
        "html", "html5", "css", "css3", "javascript", "typescript",
        "react", "react.js", "angular", "vue",
        # backend — confirmed
        "python", "php", "java", "node.js", "laravel", "django",
        ".net", "c#", "asp.net",
        # databases — confirmed
        "sql", "mysql", "postgresql", "mongodb",
        # APIs — confirmed
        "rest api", "restful apis", "api integration",
        # devops — confirmed
        "docker", "git", "aws", "ci/cd",
        # ERP — confirmed
        "erp", "crm",
        # project — confirmed
        "agile", "agile methodologies", "scrum", "project management",
        "project/program management", "full stack development",
        "software engineering", "team leadership",
    ],

    "Mobile": [
        # cross-platform — confirmed very high
        "flutter", "dart", "react native",
        # Android — confirmed
        "android", "android development",
        # iOS — confirmed
        "ios", "ios development", "swift", "swiftui", "xcode",
        "objective-c",
        # shared — confirmed
        "mobile application development", "mobile app development",
        "mobile development", "mobile developer",
        "restful apis", "apis", "firebase",
        "git", "version control (git)", "debugging",
        "ui/ux design", "application support",
        # soft — confirmed
        "agile methodologies", "software engineering",
    ],

    "Quality Control": [
        # helpdesk/support — confirmed very high
        "technical troubleshooting", "technical support",
        "customer service", "customer service/support",
        "customer support", "customer care",
        "helpdesk", "help desk", "it support",
        "it service management", "itil",
        # infra — confirmed
        "troubleshooting", "network administration", "system administration",
        "windows and mac os support", "ccna",
        # tools — confirmed
        "crm", "ticketing systems",
        # soft — confirmed
        "problem solving", "problem-solving", "time management",
        "multitasking", "english", "communication skills",
        "teamwork", "negotiation",
    ],

    "Software Tester": [
        # testing — confirmed very high
        "software testing", "quality", "testing",
        "manual testing", "test automation", "automation testing",
        "quality assurance", "quality control",
        "regression testing", "security testing",
        "test case design", "bug tracking", "defect tracking",
        # tools — confirmed
        "selenium", "cypress", "jest", "postman", "jira",
        # certifications — confirmed
        "istqb",
        # soft — confirmed
        "analytical thinking", "agile", "agile methodologies",
        "software engineering",
    ],

    "UI/UX": [
        # design tools — confirmed
        "figma", "adobe xd", "adobe photoshop", "adobe illustrator",
        "adobe indesign", "adobe creative suite", "canva",
        # disciplines — confirmed high frequency
        "ui design", "ux design", "ui/ux design", "interaction design",
        "visual design", "prototyping", "wireframing",
        "user research", "usability testing", "design systems",
        # dev collab — confirmed
        "responsive design", "css", "html", "front-end development",
        "wordpress",
        # project — confirmed
        "requirements gathering", "product management",
        "agile methodologies", "agile",
        # soft — confirmed
        "analytical thinking", "collaboration",
    ],

    "Cyber Security": [
        # confirmed in dataset
        "cyber security", "cybersecurity", "network security", "it security",
        "cybersecurity leadership", "system security", "it infrastructure",
        "penetration testing", "ethical hacking", "vulnerability scanning",
        "vulnerability assessment", "siem",
        "firewall", "encryption", "identity management",
        "microsoft active directory", "active directory",
        "linux", "kali linux", "python",
        "cloud computing", "risk management", "incident response",
        "network administration", "project management",
    ],

    "Project Manager": [
        # confirmed very high
        "project management", "project/program management", "pmp",
        "agile", "agile methodologies", "agile software", "scrum", "kanban",
        "team leadership", "budgeting and scheduling", "risk assessment",
        "jira", "ms project", "cmmi",
        "stakeholder management", "requirements gathering",
        "software engineering", "microsoft office",
        # soft — confirmed
        "leadership", "management",
    ],

    "Graphic Designer": [
        # tools — confirmed
        "adobe photoshop", "adobe illustrator", "adobe indesign",
        "adobe xd", "adobe creative suite", "adobe target",
        "figma", "canva", "after effects", "motion graphics",
        # disciplines — confirmed
        "graphic design", "ux design", "ui design", "product design",
        "social media", "digital marketing", "content writing",
        "seo", "online marketing", "marketing", "branding",
        # web — confirmed
        "e-commerce", "website optimization",
        "agile", "agile methodology",
    ],

    "Software Engineer": [
        # confirmed
        "software engineering", "agile methodologies", "agile",
        "project management", "project/program management",
        "product management", "stakeholder management",
        "sql", "azure", "erp system", "apis", "docker",
        "gitlab", "git", "devops",
        "team leadership", "leadership", "leadership management",
        "remote team collaboration", "technical support",
        "user experience (ux)",
    ],
}

WEIGHTS = {
    # AI — strongest signals
    "machine learning": 5, "deep learning": 5, "tensorflow": 5, "pytorch": 5,
    "artificial intelligence": 4, "python programming": 3,
    "natural language processing": 4, "natural language processing (nlp)": 4,
    "nlp": 4, "computer vision": 4, "neural networks": 4,
    "large language models (llms)": 5, "llm": 5, "generative ai": 5,
    "transformers": 4, "reinforcement learning": 4,
    "algorithm development": 3, "data preprocessing": 3,
    # Data
    "power bi": 4, "tableau": 4, "data analysis": 3, "data analytics": 3,
    "statistical analysis": 3, "business analysis": 3, "data science": 4,
    "business process analysis": 3,
    # Backend — ERP is very strong signal in this dataset
    "odoo development": 5, "odoo": 4, "erp implementation": 4,
    "django": 4, "laravel": 4, "fastapi": 4, ".net core": 3,
    "asp.net core": 3, "backend development": 4, "ccna": 3,
    # Frontend
    "react.js": 4, "angular": 5, "vue": 3, "vue 3": 3,
    "next.js": 4, "next14": 4, "typescript": 3,
    "front-end development": 4, "front-end": 3,
    # Mobile — strongest discriminators
    "flutter": 5, "dart": 5, "swift": 5, "kotlin": 5,
    "swiftui": 4, "react native": 4,
    "ios development": 4, "android development": 4,
    "mobile application development": 4, "mobile app development": 4,
    # QA/Testing
    "selenium": 4, "cypress": 4, "test automation": 4,
    "manual testing": 4, "software testing": 4, "quality assurance": 3,
    "istqb": 5, "defect tracking": 3, "regression testing": 3,
    # UI/UX
    "figma": 4, "adobe xd": 4, "wireframing": 4, "prototyping": 4,
    "user research": 4, "usability testing": 4, "interaction design": 4,
    "ui design": 4, "ux design": 4, "ui/ux design": 5,
    # Security
    "penetration testing": 5, "ethical hacking": 5, "siem": 5,
    "vulnerability scanning": 4, "kali linux": 4, "cybersecurity": 4,
    "cybersecurity leadership": 5,
    # PM
    "pmp": 5, "project/program management": 4, "cmmi": 4,
    "budgeting and scheduling": 3, "risk assessment": 3,
    # Graphic
    "adobe illustrator": 4, "graphic design": 4, "adobe target": 3,
    # Shared (lower weight — appear in many categories)
    "python": 2, "sql": 2, "javascript": 2, "docker": 2, "git": 2,
    "agile": 2, "agile methodologies": 2, "linux": 2, "aws": 2,
    "software engineering": 1, "team leadership": 1,
    "project management": 1, "problem solving": 1,
}

TITLE_KEYWORDS = {
    "Artificial Intelligence": ["ai","machine learning","ml engineer","deep learning","nlp","data scientist","ai engineer","llm"],
    "Data Analysis":           ["data analyst","data analysis","business analyst","bi analyst","kpi","reporting analyst"],
    "Data Science":            ["data science","data scientist","ml","machine learning"],
    "Backend":                 ["backend","back-end","back end","odoo","erp","php","java developer","python developer",".net","node","laravel"],
    "Frontend":                ["frontend","front-end","react","angular","vue","ui developer","next.js"],
    "Full Stack":              ["full stack","fullstack","full-stack"],
    "Mobile":                  ["mobile","ios","android","flutter","react native","dart"],
    "Quality Control":         ["support","helpdesk","help desk","technical support","it support","quality control","network engineer"],
    "Software Tester":         ["tester","qa","quality assurance","automation engineer","software testing","istqb"],
    "UI/UX":                   ["ui/ux","ui designer","ux designer","product designer","ui developer","interaction designer"],
    "Cyber Security":          ["security","cyber","penetration","soc analyst","cybersecurity"],
    "Project Manager":         ["project manager","program manager","scrum master","pmo","project coordinator"],
    "Graphic Designer":        ["graphic designer","graphic design","motion designer","visual designer"],
    "Software Engineer":       ["software engineer","swe","software developer"],
}

NOISE_REQ = {
    "it/software development", "engineering - telecom/technology",
    "information technology (it)", "computer science", "experienced",
    "entry level", "student", "senior management", "internship",
    "males_only", "males_preferred", "females_only",
    "full time", "part time", "freelance / project",
    "software development", "engineering", "communication",
    "problem solving", "business development", "sales/retail",
    "analyst/research", "creative/design/art",
    "installation/maintenance/repair", "operations/management",
}

ALL_SKILLS_SORTED = sorted(
    {skill for skills in CAREER_SKILLS.values() for skill in skills},
    key=len, reverse=True,
)

# Short skills (<=2 chars) need word-boundary matching to avoid false positives.
# e.g. "r" must NOT match inside "for", "pytorch", "presenter", "cairo", "years"
_SHORT_SKILL_PATTERNS = {
    skill: re.compile(r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])")
    for skill in ALL_SKILLS_SORTED if len(skill) <= 2
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING & ENRICHMENT
# ─────────────────────────────────────────────────────────────

def parse_requirements(req: str) -> str:
    if pd.isna(req):
        return ""
    parts = [p.strip().lower() for p in req.split("·")]
    cleaned = []
    for p in parts:
        if not p or len(p) < 3: continue
        if re.match(r"\d", p): continue
        if p in NOISE_REQ: continue
        if re.search(r"\d+\s*(yrs?|years?)", p): continue
        cleaned.append(p)
    return " ".join(cleaned)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["cleaned_skills_text_v2", "Speciality"])
    df["cleaned_skills_text_v2"] = df["cleaned_skills_text_v2"].str.lower().str.strip()
    df["Speciality"] = df["Speciality"].str.strip()
    df = df.reset_index(drop=True)
    df["req_parsed"] = df["Requirement"].apply(parse_requirements)
    df["enriched_text"] = (df["cleaned_skills_text_v2"] + " " + df["req_parsed"]).str.strip()
    return df


# Synonym map — normalise variant spellings before vectorising
SYNONYMS = {
    "js":                     "javascript",
    "ts":                     "typescript",
    "py":                     "python",
    "reactjs":                "react.js",
    "react js":               "react.js",
    "nodejs":                 "node.js",
    "node js":                "node.js",
    "vuejs":                  "vue",
    "vue.js":                 "vue",
    "angularjs":              "angular",
    "dotnet":                 ".net",
    "dot net":                ".net",
    "asp net":                "asp.net",
    "c sharp":                "c#",
    "postgres":               "postgresql",
    "mongo":                  "mongodb",
    "mongo db":               "mongodb",
    "mysql server":           "mysql",
    "ms sql server":          "sql server",
    "power-bi":               "power bi",
    "powerbi":                "power bi",
    "tableau desktop":        "tableau",
    "scikit learn":           "scikit-learn",
    "sklearn":                "scikit-learn",
    "hugging face":           "huggingface",
    "llms":                   "llm",
    "large language model":   "llm",
    "generative artificial intelligence": "generative ai",
    "ui ux":                  "ui/ux",
    "ux ui":                  "ui/ux",
    "figma design":           "figma",
    "adobe photoshop":        "adobe photoshop",
    "photoshop":              "adobe photoshop",
    "illustrator":            "adobe illustrator",
    "react-native":           "react native",
    "flutter dart":           "flutter",
    "ios swift":              "swift",
    "android kotlin":         "kotlin",
    "nlp":                    "nlp",
    "natural language processing (nlp)": "nlp",
    "erp systems":            "erp",
    "erp software":           "erp",
    "git github":             "git",
    "version control":        "git",
    "version control (git)":  "git",
    "restful api":            "restful apis",
    "rest apis":              "restful apis",
    "rest api":               "restful apis",
    "agile scrum":            "agile methodologies",
    "agile methodology":      "agile methodologies",
    "istqb certified":        "istqb",
    "test automation":        "test automation",
    "automated testing":      "test automation",
    "manual testing":         "manual testing",
    "manual test":            "manual testing",
    "quality assurance (qa)": "quality assurance",
    "qa testing":             "quality assurance",
    "help desk":              "helpdesk",
    "it helpdesk":            "helpdesk",
    "technical troubleshoot":"technical troubleshooting",
    "customer service/support":"customer service",
    "project/program management":"project management",
    "budgeting and scheduling":"budgeting",
}

def apply_synonyms(text: str) -> str:
    """Normalise variant spellings to canonical skill names."""
    for variant, canonical in SYNONYMS.items():
        text = re.sub(r'(?<![a-z])' + re.escape(variant) + r'(?![a-z])', canonical, text)
    return text


def build_vectorizer(df: pd.DataFrame):
    df["enriched_text"] = df["enriched_text"].apply(apply_synonyms)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),      
        max_features=12000,
        sublinear_tf=True,      
        min_df=2,                
        max_df=0.95,             
    )
    job_vectors = vectorizer.fit_transform(df["enriched_text"])
    centroids  = {}
    intra_avg  = {}   
    for spec in df["Speciality"].unique():
        idx      = df.index[df["Speciality"] == spec].tolist()
        centroid = np.asarray(job_vectors[idx].mean(axis=0))
        centroids[spec]  = centroid
        sims             = cosine_similarity(centroid, job_vectors[idx])[0]
        intra_avg[spec]  = float(sims.mean())
    return vectorizer, job_vectors, centroids, intra_avg

# ─────────────────────────────────────────────────────────────
# TEXT & SKILL PROCESSING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b\d{4}\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^\w\s./#+\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text: str) -> list:
    text = text.lower()
    found = set()
    consumed = text
    for skill in ALL_SKILLS_SORTED:
        if skill in _SHORT_SKILL_PATTERNS:
            if _SHORT_SKILL_PATTERNS[skill].search(consumed):
                found.add(skill)
                consumed = _SHORT_SKILL_PATTERNS[skill].sub(" " * len(skill), consumed)
        else:
            if skill in consumed:
                found.add(skill)
                consumed = consumed.replace(skill, " " * len(skill))
    return sorted(found)


def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def process_cv(filepath: str):
    raw = extract_text_from_pdf(filepath)
    cleaned = clean_text(raw)

    # Normalise variant spellings before skill extraction
    normalised = apply_synonyms(cleaned)
    skills = extract_skills(normalised)

    # Boost high-signal skills 5x, normal skills 3x
    high_signal   = [s for s in skills if WEIGHTS.get(s, 1) >= 4]
    normal_signal = [s for s in skills if WEIGHTS.get(s, 1) < 4]
    skill_boost   = " ".join(high_signal * 5 + normal_signal * 3)
    boosted       = normalised + " " + skill_boost

    return boosted, skills, normalised

def process_text(user_message: str):
    """
    Process a plain-text message from the user instead of a PDF.
    Accepts natural language like:
      "I am a frontend developer, my skills are react, typescript, angular and figma"
      "python machine learning tensorflow deep learning nlp"
      "i work as mobile developer flutter dart ios swift"
    Returns (boosted_text, skills, normalised_text) — same shape as process_cv()
    """
    cleaned    = clean_text(user_message)
    normalised = apply_synonyms(cleaned)
    skills     = extract_skills(normalised)

    # If user explicitly mentioned a career/role, inject its skills as extra signal
    role_boost = []
    text_lower = normalised.lower()
    for career, keywords in TITLE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            # Add all skills for that career once as soft signal
            role_boost.extend(CAREER_SKILLS.get(career, []))
            break  # only boost the first matched role

    all_skills     = list(set(skills + extract_skills(" ".join(role_boost))))
    high_signal    = [s for s in all_skills if WEIGHTS.get(s, 1) >= 4]
    normal_signal  = [s for s in all_skills if WEIGHTS.get(s, 1) < 4]
    skill_boost    = " ".join(high_signal * 5 + normal_signal * 3)
    boosted        = normalised + " " + skill_boost

    return boosted, skills, normalised


def recommend_from_text(
    user_message: str,
    df,
    vectorizer,
    job_vectors,
    centroids: dict,
    intra_avg: dict,
    top_n: int = 5,
) -> dict:
    """
    Same pipeline as recommend() but takes a plain-text message instead of a PDF file.
    """
    cv_boosted, cv_skills, cv_raw = process_text(user_message)
    career, confidence, top3 = detect_career(cv_raw, vectorizer, centroids, intra_avg)

    filtered = df[df["Speciality"] == career].reset_index(drop=True)
    fallback = len(filtered) < top_n
    if fallback:
        filtered = df.reset_index(drop=True)

    fv      = vectorizer.transform(filtered["enriched_text"])
    cv_vec  = vectorizer.transform([cv_boosted])
    sims    = cosine_similarity(cv_vec, fv)[0]
    top_idx = sims.argsort()[-top_n:][::-1]

    jobs = []
    for idx in top_idx:
        row = filtered.iloc[idx]
        gap = skill_gap(cv_skills, row["enriched_text"])
        jobs.append({
            "title":             row["Job Title"],
            "company":           row["Company Name"].strip(),
            "location":          row["Company Location"],
            "job_location_type": row["Job Location"],
            "job_type":          row["Job Time"],
            "url":               row["Job URL"],
            "speciality":        row["Speciality"],
            "similarity":        round(float(sims[idx]) * 100, 1),
            "skill_match_rate":  gap["match_rate"],
            "matched_skills":    gap["matched"],
            "missing_skills":    gap["missing"],
            "total_required":    gap["total_required"],
        })

    return {
        "detected_career": career,
        "confidence":      confidence,
        "top3_careers":    [{"career": c, "score": round(s * 100, 1)} for c, s in top3],
        "cv_skills":       cv_skills,
        "total_cv_skills": len(cv_skills),
        "fallback_used":   fallback,
        "input_type":      "text",
        "jobs":            jobs,
    }


# ─────────────────────────────────────────────────────────────
# CAREER DETECTION
# ─────────────────────────────────────────────────────────────

def detect_career(cv_text: str, vectorizer, centroids: dict, intra_avg: dict):
    """
    Hybrid career detection: keyword score + normalised centroid similarity + title bonus.
    Centroid similarity is divided by the intra-cluster average so that tight small
    categories (e.g. AI with 22 jobs) don't dominate large ones (e.g. Frontend, 30 jobs).
    """
    text = cv_text.lower()
    careers = list(CAREER_SKILLS.keys())

    # A: weighted keyword score
    kw_scores = {
        c: sum(WEIGHTS.get(s, 1) for s in skills if s in text)
        for c, skills in CAREER_SKILLS.items()
    }
    kw_max = max(kw_scores.values()) or 1
    kw_norm = {c: v / kw_max for c, v in kw_scores.items()}

    # B: centroid similarity — normalised by intra-cluster avg
    # raw_sim / intra_avg: score > 1 means CV fits better than the average job in that cluster
    cv_vec = vectorizer.transform([cv_text])
    sim_scores = {}
    for c in careers:
        if c in centroids:
            raw_sim = float(cosine_similarity(cv_vec, centroids[c])[0][0])
            sim_scores[c] = raw_sim / (intra_avg.get(c, 1.0) + 1e-6)
        else:
            sim_scores[c] = 0.0
    sim_max = max(sim_scores.values()) or 1
    sim_norm = {c: v / sim_max for c, v in sim_scores.items()}

    title_bonus = {c: 0.0 for c in careers}
    for career, keywords in TITLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            title_bonus[career] = 1.0
    tb_max = max(title_bonus.values()) or 1
    tb_norm = {c: v / tb_max for c, v in title_bonus.items()}

    combined = {
        c: 0.40 * kw_norm[c] + 0.40 * sim_norm[c] + 0.20 * tb_norm[c]
        for c in careers
    }
    best = max(combined, key=combined.get)

    vals = np.array([combined[c] for c in careers])
    exp_v = np.exp(vals - vals.max())
    confidence = round(float(exp_v[careers.index(best)] / exp_v.sum()) * 100, 1)

    top3 = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]
    return best, confidence, top3

# ─────────────────────────────────────────────────────────────
# SKILL GAP ANALYSIS
# ─────────────────────────────────────────────────────────────

def skill_gap(cv_skills: list, job_enriched: str) -> dict:
    job_skills = set(extract_skills(job_enriched))
    cv_set = set(cv_skills)
    matched = sorted(cv_set & job_skills)
    missing = sorted(job_skills - cv_set)
    total = len(job_skills)
    return {
        "matched": matched,
        "missing": missing,
        "match_rate": round(len(matched) / (total + 1e-5) * 100, 1),
        "total_required": total,
    }

# ─────────────────────────────────────────────────────────────
# MAIN RECOMMEND FUNCTION
# ─────────────────────────────────────────────────────────────

def recommend(
    cv_filepath: str,
    df: pd.DataFrame,
    vectorizer,
    job_vectors,
    centroids: dict,
    intra_avg: dict,
    top_n: int = 5,
) -> dict:
    """
    Returns a dict with:
      - detected_career, confidence, top3_careers
      - cv_skills
      - jobs: list of dicts with title, company, location, job_type, url,
              similarity, skill_match_rate, matched_skills, missing_skills
    """
    cv_boosted, cv_skills, cv_raw = process_cv(cv_filepath)
    career, confidence, top3 = detect_career(cv_raw, vectorizer, centroids, intra_avg)

    filtered = df[df["Speciality"] == career].reset_index(drop=True)
    fallback = len(filtered) < top_n
    if fallback:
        filtered = df.reset_index(drop=True)

    fv = vectorizer.transform(filtered["enriched_text"])
    cv_vec = vectorizer.transform([cv_boosted])
    sims = cosine_similarity(cv_vec, fv)[0]
    top_idx = sims.argsort()[-top_n:][::-1]

    jobs = []
    for idx in top_idx:
        row = filtered.iloc[idx]
        gap = skill_gap(cv_skills, row["enriched_text"])
        jobs.append({
            "title":            row["Job Title"],
            "company":          row["Company Name"].strip(),
            "location":         row["Company Location"],
            "job_location_type": row["Job Location"],
            "job_type":         row["Job Time"],
            "url":              row["Job URL"],
            "speciality":       row["Speciality"],
            "similarity":       round(float(sims[idx]) * 100, 1),
            "skill_match_rate": gap["match_rate"],
            "matched_skills":   gap["matched"],
            "missing_skills":   gap["missing"],
            "total_required":   gap["total_required"],
        })

    return {
        "detected_career": career,
        "confidence":      confidence,
        "top3_careers":    [{"career": c, "score": round(s * 100, 1)} for c, s in top3],
        "cv_skills":       cv_skills,
        "total_cv_skills": len(cv_skills),
        "fallback_used":   fallback,
        "jobs":            jobs,
    }
