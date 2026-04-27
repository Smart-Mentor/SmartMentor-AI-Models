import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

# =========================
# LOAD DATA
# =========================
skills_path = r"D:\Games\AI MODELS\Models\RoadMap Ai\DataSets\skills_roadmap.csv"
train_path = r"D:\Games\AI MODELS\Models\RoadMap Ai\DataSets\training_data22.csv"

skills_df = pd.read_csv(skills_path)
train_df = pd.read_csv(train_path)

# =========================
# NORMALIZATION
# =========================
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

for col in skills_df.columns:
    skills_df[col] = skills_df[col].astype(str).map(normalize)

for col in train_df.columns:
    train_df[col] = train_df[col].astype(str).map(normalize)

skills_df["duration_weeks"] = pd.to_numeric(
    skills_df["duration_weeks"], errors="coerce"
).fillna(1)

# =========================
# NLP MODEL
# =========================
print("🔄 Loading NLP model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

subjects = skills_df["subject"].unique().tolist()
skills_list = skills_df["skill"].unique().tolist()

subject_emb = model.encode(subjects, convert_to_tensor=True)
skill_emb = model.encode(skills_list, convert_to_tensor=True)

# =========================
# SUBJECT MAPPING (STRICT)
# =========================
def map_subject(subject):
    """Exact mapping for known career paths"""
    mapping = {
        "ai engineer": "ai engineer",
        "ai": "ai engineer",
        "artificial intelligence": "ai engineer",
        "ml engineer": "ai engineer",
        "machine learning engineer": "ai engineer",
        "data scientist": "data scientist",
        "data science": "data scientist",
        "frontend developer": "frontend developer",
        "frontend": "frontend developer",
        "backend developer": "backend developer",
        "backend": "backend developer",
        "devops engineer": "devops engineer",
        "devops": "devops engineer",
        "mobile developer": "mobile developer",
        "mobile": "mobile developer",
        "cybersecurity": "cybersecurity",
        "security": "cybersecurity",
        "game developer": "game developer",
        "game dev": "game developer",
        "data engineer": "data engineer"
    }
    
    subject_lower = subject.lower().strip()
    
    # Check exact mapping first
    if subject_lower in mapping:
        return mapping[subject_lower]
    
    # Check if subject contains any mapping key
    for key, value in mapping.items():
        if key in subject_lower:
            return value
    
    return subject

def find_closest_subject(user_subject):
    """Fallback for unrecognized subjects"""
    emb = model.encode(user_subject, convert_to_tensor=True)
    scores = util.cos_sim(emb, subject_emb)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    
    # Only use semantic matching if confidence is high
    if best_score > 0.7:
        return subjects[best_idx]
    return None

# =========================
# NLP EXTRACTION
# =========================
def extract(text):
    text = normalize(text)

    if not text:
        return None, []

    detected_skills = []

    # EXACT MATCH for skills
    for s in skills_list:
        if re.search(r"\b" + re.escape(s) + r"\b", text):
            detected_skills.append(s)

    emb = model.encode(text, convert_to_tensor=True)

    # SUBJECT - try to extract from common patterns first
    subject = None
    
    # Priority: Look for career mentions in the text
    career_patterns = {
        "ai engineer": ["ai engineer", "ai engineering", "machine learning engineer"],
        "data scientist": ["data scientist", "data science"],
        "frontend developer": ["frontend", "front-end", "front end developer"],
        "backend developer": ["backend", "back-end", "back end developer"],
        "devops engineer": ["devops"],
        "mobile developer": ["mobile developer", "mobile dev", "ios developer", "android developer"],
        "cybersecurity": ["cybersecurity", "cyber security", "security engineer"],
        "game developer": ["game developer", "game dev", "game programmer"],
        "data engineer": ["data engineer"]
    }
    
    text_lower = text.lower()
    for career, patterns in career_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                subject = career
                break
        if subject:
            break
    
    # If no career pattern found, use semantic matching
    if not subject:
        sub_scores = util.cos_sim(emb, subject_emb)[0]
        best_idx = int(np.argmax(sub_scores))
        best_score = float(sub_scores[best_idx])

        if best_score > 0.65:
            subject = subjects[best_idx]

    # SKILLS (semantic)
    skill_scores = util.cos_sim(emb, skill_emb)[0]

    for i, score in enumerate(skill_scores):
        if float(score) > 0.65:  # Increased threshold
            detected_skills.append(skills_list[i])

    detected_skills = list(set(detected_skills))

    return subject, detected_skills

# =========================
# FILTER SKILLS
# =========================
def filter_skills(skills, subject):
    valid_skills = set(skills_df[skills_df["subject"] == subject]["skill"])
    
    # Also include prerequisite skills that might be needed
    filtered = []
    for s in skills:
        if s in valid_skills:
            filtered.append(s)
        else:
            # Check if this skill exists as a prerequisite in this subject
            df_subject = skills_df[skills_df["subject"] == subject]
            if any(s in str(prereq) for prereq in df_subject["prerequisite"]):
                filtered.append(s)
    
    return filtered

# =========================
# BUILD SKILL PRIORITY
# =========================
def build_skill_priority():
    priority = {}

    weight_map = {
        "beginner path": 3,
        "fast track": 2,
        "project based": 1
    }

    for _, row in train_df.iterrows():
        roadmap = row.get("preferred_roadmap", "")
        skills_str = row.get("skills", "")

        if not skills_str:
            continue

        skills_tokens = re.split(r",|and|\s+", skills_str)
        skills_tokens = [s.strip() for s in skills_tokens if s.strip()]

        for skill in skills_tokens:
            if skill not in priority:
                priority[skill] = 0

            priority[skill] += weight_map.get(roadmap, 1)

    return priority

skill_priority = build_skill_priority()

# =========================
# BUILD SEQUENCE
# =========================
def build_sequence(subject):
    df = skills_df[skills_df["subject"] == subject].copy()

    if df.empty:
        return df

    level_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
    df["level_rank"] = df["level"].map(level_order).fillna(99)

    df["priority"] = df["skill"].map(lambda x: skill_priority.get(x, 0))

    # Sort by level first, then priority
    df = df.sort_values(
        by=["level_rank", "priority"],
        ascending=[True, False]
    )

    return df

# =========================
# GENERATE ROADMAP
# =========================
def generate_roadmap(subject, user_skills):
    df = build_sequence(subject)
    
    if df.empty:
        return []

    roadmap = []
    learned = set(user_skills)
    
    # Keep track of added skills to avoid duplicates
    added_skills = set()

    # First pass: Add skills that directly match user's input
    for _, row in df.iterrows():
        skill = row["skill"]
        
        if skill in learned and skill not in added_skills:
            roadmap.append({
                "skill": skill,
                "level": row["level"],
                "duration": int(row["duration_weeks"]),
                "priority": row["priority"],
                "reason": "Already knows this skill"
            })
            added_skills.add(skill)
    
    # Second pass: Add prerequisites and build the rest
    max_iterations = len(df)
    for _ in range(max_iterations):
        added_this_round = False
        
        for _, row in df.iterrows():
            skill = row["skill"]
            
            if skill in added_skills:
                continue
                
            prereq = row["prerequisite"]
            
            # Skip if prerequisite is "none"
            if prereq == "none" or prereq == "None":
                prereqs = []
            else:
                prereqs = [p.strip() for p in str(prereq).split(",") if p.strip() and p.strip() != "none"]
            
            # Check if all prerequisites are either learned or already in roadmap
            prereqs_satisfied = True
            missing_prereqs = []
            
            for p in prereqs:
                if p not in learned and p not in added_skills:
                    prereqs_satisfied = False
                    missing_prereqs.append(p)
            
            if prereqs_satisfied:
                roadmap.append({
                    "skill": skill,
                    "level": row["level"],
                    "duration": int(row["duration_weeks"]),
                    "priority": row["priority"],
                    "reason": f"Prerequisites satisfied (needs: {', '.join(missing_prereqs) if missing_prereqs else 'none'})"
                })
                added_skills.add(skill)
                added_this_round = True
        
        if not added_this_round:
            break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_roadmap = []
    for item in roadmap:
        if item["skill"] not in seen:
            seen.add(item["skill"])
            unique_roadmap.append(item)
    
    return unique_roadmap

# =========================
# TRACK DETECTION
# =========================
def detect_track(skills, subject):
    df = skills_df[skills_df["subject"] == subject]

    if df.empty:
        return "Beginner Path"

    required = set(df["skill"])

    if not skills:
        return "Beginner Path"

    coverage = len(set(skills) & required) / len(required)

    if coverage < 0.25:
        return "Beginner Path"
    elif coverage < 0.65:
        return "Fast Track"
    return "Project-Based"

# =========================
# MAIN
# =========================
def run():
    print("\n" + "="*50)
    print("📚 ROADMAP GENERATOR")
    print("="*50)
    print("\n💬 Describe your goal (e.g., 'I want to become an AI Engineer' or 'AI Engineer with Python and NLP'):")
    user = input("You: ")

    subject, skills = extract(user)

    # ===== HANDLE INPUT =====
    if not subject and not skills:
        print("\n⚠️ Couldn't detect anything. Let me help you...")
        print("\nAvailable careers:")
        print(", ".join(subjects))
        subject = normalize(input("\n🎯 Enter career name: "))
        skills_input = input("🔧 Enter skills you already know (comma-separated, or 'none'): ").lower()

        if skills_input != "none":
            skills = re.split(r",|and", skills_input)
            skills = [normalize(s.strip()) for s in skills if s.strip()]
        else:
            skills = []

    elif skills and not subject:
        print(f"\n🧠 Skills detected: {skills}")
        print("\nAvailable careers:")
        print(", ".join(subjects))
        subject = normalize(input("\n🎯 Enter career name: "))

    elif subject and not skills:
        print(f"\n🎯 Career detected: {subject}")
        skills_input = input("🔧 Enter skills you already know (comma-separated, or 'none'): ").lower()

        if skills_input != "none":
            skills = re.split(r",|and", skills_input)
            skills = [normalize(s.strip()) for s in skills if s.strip()]
        else:
            skills = []

    else:
        print(f"\n🎯 Career: {subject}")
        print(f"🧠 Skills: {skills}")

    # ===== FIX SUBJECT =====
    original_subject = subject
    subject = map_subject(subject)

    if subject not in subjects:
        new_subject = find_closest_subject(subject)
        if new_subject:
            print(f"🔁 Mapped '{original_subject}' to: {new_subject}")
            subject = new_subject
        else:
            print(f"\n❌ Career '{original_subject}' not recognized.")
            print("Available careers:")
            print(", ".join(subjects))
            return

    # ===== FILTER SKILLS =====
    skills = filter_skills(skills, subject)
    
    if skills:
        print(f"\n✅ Valid skills for {subject}: {skills}")
    else:
        print(f"\nℹ️ No specific skills detected for {subject}")

    # ===== VALIDATION =====
    if subject not in skills_df["subject"].unique():
        print(f"\n❌ Career '{subject}' not found in database.")
        return

    # ===== GENERATE =====
    track = detect_track(skills, subject)
    roadmap = generate_roadmap(subject, skills)

    if not roadmap:
        print(f"\n⚠️ No roadmap found for {subject}.")
        print("Please check if the career exists in the database.")
        return

    # ===== OUTPUT =====
    print("\n" + "="*50)
    print(f"🚀 CAREER PATH: {subject.upper()}")
    print(f"📊 TRACK: {track}")
    print("="*50)
    print("\n📋 YOUR LEARNING ROADMAP:\n")

    total_time = 0

    for i, step in enumerate(roadmap, 1):
        known_flag = "✓" if step["skill"] in skills else "→"
        print(f"{known_flag} {i}. {step['skill'].title()} ({step['level']}, {step['duration']} weeks)")

    print("\n" + "-"*50)
    for step in roadmap:
        if step["skill"] in skills:
            total_time += 0
        else:
            total_time += step["duration"]

    print(f"\n⏱️ Estimated time to complete: {total_time} weeks")
    
    if total_time == 0:
        print("🎉 Congratulations! You already know all the skills needed!")
    
    print("="*50)

# =========================
# LOOP
# =========================
if __name__ == "__main__":
    while True:
        run()
        if input("\n🔄 Generate another roadmap? (y/n): ").lower() != "y":
            print("\n👋 Good luck with your learning journey!")
            break