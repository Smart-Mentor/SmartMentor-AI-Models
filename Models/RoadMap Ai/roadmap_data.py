# # roadmap_data = {
# #     "frontend": {
# #         "HTML": {
# #             "prerequisites": [],
# #             "courses": ["https://www.youtube.com/watch?v=qz0aGYrrlhU"]
# #         },
# #         "CSS": {
# #             "prerequisites": ["HTML"],
# #             "courses": ["https://www.youtube.com/watch?v=1PnVor36_40"]
# #         },
# #         "JavaScript": {
# #             "prerequisites": ["HTML", "CSS"],
# #             "courses": ["https://www.youtube.com/watch?v=W6NZfCO5SIk"]
# #         },
# #         "React": {
# #             "prerequisites": ["JavaScript"],
# #             "courses": ["https://www.youtube.com/watch?v=bMknfKXIFA8"]
# #         }
# #     },

# #     "backend": {
# #         "Python": {
# #             "prerequisites": [],
# #             "courses": ["https://www.youtube.com/watch?v=_uQrJ0TkZlc"]
# #         },
# #         "Flask": {
# #             "prerequisites": ["Python"],
# #             "courses": ["https://www.youtube.com/watch?v=Z1RJmh_OqeA"]
# #         },
# #         "SQL": {
# #             "prerequisites": ["Python"],
# #             "courses": ["https://www.youtube.com/watch?v=HXV3zeQKqGY"]
# #         }
# #     },

# #     "ai": {
# #         "Python": {
# #             "prerequisites": [],
# #             "courses": ["https://www.youtube.com/watch?v=_uQrJ0TkZlc"]
# #         },
# #         "Numpy": {
# #             "prerequisites": ["Python"],
# #             "courses": ["https://www.youtube.com/watch?v=QUT1VHiLmmI"]
# #         },
# #         "Machine Learning": {
# #             "prerequisites": ["Numpy"],
# #             "courses": ["https://www.youtube.com/watch?v=Gv9_4yMHFhI"]
# #         }
# #     }
# # }

# # import pandas as pd
# # import re
# # import random
# # import json
# # import os
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.ensemble import RandomForestClassifier

# # # =========================
# # # LOAD DATA
# # # =========================
# # skills_df = pd.read_csv("D:/Games/RoadMap Ai/skills_roadmap.csv")
# # train_df = pd.read_csv("D:/Games/RoadMap Ai/training_data22.csv")

# # # =========================
# # # NORMALIZATION
# # # =========================
# # def normalize(text):
# #     if pd.isna(text) or text is None:
# #         return ""
# #     return str(text).strip().lower()

# # skills_df["skill"] = skills_df["skill"].apply(normalize)
# # skills_df["prerequisite"] = skills_df["prerequisite"].apply(normalize)
# # skills_df["subject"] = skills_df["subject"].apply(normalize)

# # train_df["skills"] = train_df["skills"].apply(normalize)
# # train_df["preferred_roadmap"] = train_df["preferred_roadmap"].fillna("Beginner Path")

# # # =========================
# # # RL SETUP
# # # =========================
# # Q_FILE = "q_table.json"
# # actions = ["Beginner Path", "Fast Track", "Project-Based"]

# # if os.path.exists(Q_FILE):
# #     try:
# #         with open(Q_FILE, "r") as f:
# #             Q = json.load(f)
# #         if not isinstance(Q, dict):
# #             Q = {}
# #     except:
# #         print("⚠️ Q-table corrupted. Resetting...")
# #         Q = {}
# # else:
# #     Q = {}

# # def save_q():
# #     with open(Q_FILE, "w") as f:
# #         json.dump(Q, f, indent=4)

# # def get_state(skills, gaps):
# #     return f"{len(skills)}_{len(gaps)}"

# # def choose_action(state, ml_suggestion, epsilon=0.2):
# #     if state not in Q:
# #         Q[state] = {a: 0 for a in actions}

# #     if random.random() < epsilon:
# #         return random.choice(actions)

# #     scores = {}
# #     for a in actions:
# #         rl_score = Q[state][a]
# #         ml_bonus = 1 if a == ml_suggestion else 0
# #         scores[a] = rl_score + ml_bonus

# #     return max(scores, key=scores.get)

# # def update_q(state, action, reward, alpha=0.1):
# #     if state not in Q:
# #         Q[state] = {a: 0 for a in actions}

# #     Q[state][action] += alpha * (reward - Q[state][action])
# #     save_q()

# # # =========================
# # # GRAPH
# # # =========================
# # def build_graph(subject):
# #     df = skills_df[skills_df["subject"] == subject]
# #     graph = {}

# #     for _, row in df.iterrows():
# #         prereq = row["prerequisite"]
# #         if prereq in ["none", ""]:
# #             continue
# #         graph.setdefault(prereq, []).append(row["skill"])

# #     return graph

# # def get_paths(graph, start, path=None):
# #     if path is None:
# #         path = [start]

# #     if start not in graph:
# #         return [path]

# #     paths = []
# #     for nxt in graph[start]:
# #         if nxt not in path:
# #             paths += get_paths(graph, nxt, path + [nxt])

# #     return paths if paths else [path]

# # def get_roadmaps(graph, start):
# #     paths = get_paths(graph, start)

# #     return {
# #         "Beginner Path": max(paths, key=len),
# #         "Fast Track": min(paths, key=len),
# #         "Project-Based": next((p for p in paths if "projects" in p), paths[0])
# #     }

# # # =========================
# # # ML MODEL
# # # =========================
# # vectorizer = TfidfVectorizer()
# # X = vectorizer.fit_transform(train_df["skills"])
# # y = train_df["preferred_roadmap"]

# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # model.fit(X, y)

# # def predict(skills):
# #     return model.predict(vectorizer.transform([" ".join(skills)]))[0]

# # # =========================
# # # SKILL LOGIC
# # # =========================
# # def get_required(subject):
# #     return set(skills_df[skills_df["subject"] == subject]["skill"])

# # def get_gaps(skills, subject):
# #     return list(get_required(subject) - set(skills))

# # def skill_based_track(skills, subject):
# #     required = get_required(subject)
# #     coverage = len(set(skills) & required) / max(len(required), 1)

# #     if coverage < 0.3:
# #         return "Beginner Path"
# #     elif coverage < 0.7:
# #         return "Fast Track"
# #     return "Project-Based"

# # # =========================
# # # NLP EXTRACTION
# # # =========================
# # career_aliases = {
# #     "ai": "ai engineer",
# #     "machine learning": "ai engineer",
# #     "frontend": "frontend developer",
# #     "web": "frontend developer",
# #     "data science": "data scientist"
# # }

# # def extract_career_and_skills(text):
# #     text = normalize(text)

# #     skills = []
# #     for s in skills_df["skill"].unique():
# #         if re.search(r'\b' + re.escape(s) + r'\b', text):
# #             skills.append(s)

# #     career = None
# #     for c in skills_df["subject"].unique():
# #         if re.search(r'\b' + re.escape(c) + r'\b', text):
# #             career = c

# #     for k, v in career_aliases.items():
# #         if k in text:
# #             career = v

# #     return career, list(set(skills))

# # def extract_skills_from_cv(text):
# #     text = normalize(text)
# #     return [s for s in skills_df["skill"].unique() if s in text]

# # # =========================
# # # INTENT
# # # =========================
# # def wants_help(text):
# #     keywords = ["help", "recommend", "best", "suggest", "choose"]
# #     return any(k in text.lower() for k in keywords)

# # # =========================
# # # SESSION MEMORY
# # # =========================
# # session = {
# #     "subject": None,
# #     "skills": []
# # }

# # # =========================
# # # CHATBOT CORE
# # # =========================
# # def run_chatbot(subject, skills, help_mode):
# #     graph = build_graph(subject)
# #     if not graph:
# #         print("❌ Invalid subject")
# #         return

# #     gaps = get_gaps(skills, subject)
# #     state = get_state(skills, gaps)

# #     start = next((s for s in skills if s in graph), list(graph.keys())[0])
# #     roadmaps = get_roadmaps(graph, start)

# #     skill_track = skill_based_track(skills, subject)
# #     path = roadmaps[skill_track]
# #     final_track = skill_track

# #     # 🤖 AI MODE (ML + RL)
# #     if help_mode:
# #         ml_suggestion = predict(skills)
# #         print(f"\n🤖 ML Suggests: {ml_suggestion}")

# #         tried = set()

# #         while True:
# #             available = [a for a in actions if a not in tried]
# #             if not available:
# #                 print("⚠️ No more options.")
# #                 break

# #             action = choose_action(state, ml_suggestion)

# #             if action not in available:
# #                 action = random.choice(available)

# #             tried.add(action)

# #             print(f"\n🤖 AI Suggestion: {action}")
# #             print("\n📚 Path:")
# #             for i, s in enumerate(roadmaps[action], 1):
# #                 print(f"{i}. {s}")

# #             fb = input("\nAccept this roadmap? (y/n): ").lower()

# #             if fb == "y":
# #                 update_q(state, action, 1)
# #                 path = roadmaps[action]
# #                 final_track = action
# #                 break
# #             else:
# #                 update_q(state, action, -1)
# #                 print("\n🔄 Trying another option...")

# #     # =========================
# #     # OUTPUT
# #     # =========================
# #     print("\n" + "="*50)
# #     print("🎯 YOUR ROADMAP")
# #     print("="*50)

# #     print(f"\n📌 Track: {final_track}")

# #     print("\n📊 Skill Gaps:")
# #     if gaps:
# #         for g in gaps:
# #             print("-", g)
# #     else:
# #         print("✅ No gaps")

# #     print("\n📚 Steps:")
# #     for i, s in enumerate(path, 1):
# #         print(f"{i}. {s}")

# # # =========================
# # # CHAT LOOP
# # # =========================
# # def chat():
# #     print("\n🤖 AI Roadmap Assistant (type 'exit' to quit)\n")

# #     while True:
# #         user = input("You: ").lower()

# #         if user == "exit":
# #             print("👋 Goodbye!")
# #             break

# #         if user == "cv":
# #             cv = input("Paste CV:\n")
# #             session["skills"] = extract_skills_from_cv(cv)
# #             print("✅ Skills updated:", session["skills"])
# #             continue

# #         subject, skills = extract_career_and_skills(user)

# #         if subject:
# #             session["subject"] = subject

# #         if skills:
# #             session["skills"] = list(set(session["skills"] + skills))

# #         if not session["subject"]:
# #             print("🎯 What career do you want?")
# #             continue

# #         if not session["skills"]:
# #             print("🔧 Tell me your skills")
# #             continue

# #         help_mode = wants_help(user)

# #         run_chatbot(session["subject"], session["skills"], help_mode)

# # # =========================
# # # RUN
# # # =========================
# # if __name__ == "__main__":
# #     chat()

# import pandas as pd
# import re
# import random
# import json
# import os
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# # =========================
# # CONFIGURATION
# # =========================
# Q_FILE = "q_table.json"
# MODEL_FILE = "roadmap_model.pkl"
# VECTORIZER_FILE = "vectorizer.pkl"
# EPSILON_START = 0.2
# EPSILON_MIN = 0.05
# EPSILON_DECAY = 0.99
# ALPHA = 0.1
# EPISODE_COUNT = 0

# # =========================
# # LOAD DATA
# # =========================
# def load_data():
#     """Load and validate datasets"""
#     try:
#         skills_df = pd.read_csv(r"D:/Games/RoadMap Ai/skills_roadmap.csv")
#         train_df = pd.read_csv(r"D:/Games/RoadMap Ai/training_data22.csv")
#         print("✅ Data loaded successfully")
#         return skills_df, train_df
#     except FileNotFoundError as e:
#         print(f"❌ Error loading data: {e}")
#         print("Please check file paths")
#         exit(1)

# skills_df, train_df = load_data()

# # =========================
# # NORMALIZATION
# # =========================
# def normalize(text):
#     """Normalize text for consistent matching"""
#     if pd.isna(text) or text is None:
#         return ""
#     # Remove special characters and extra spaces
#     text = str(text).strip().lower()
#     text = re.sub(r'[^\w\s]', ' ', text)
#     return ' '.join(text.split())

# skills_df["skill"] = skills_df["skill"].apply(normalize)
# skills_df["prerequisite"] = skills_df["prerequisite"].apply(normalize)
# skills_df["subject"] = skills_df["subject"].apply(normalize)

# train_df["skills"] = train_df["skills"].apply(normalize)
# train_df["preferred_roadmap"] = train_df["preferred_roadmap"].fillna("Beginner Path")

# # =========================
# # RL SETUP WITH DECAY
# # =========================
# actions = ["Beginner Path", "Fast Track", "Project-Based"]

# def load_q_table():
#     """Load Q-table from file with validation"""
#     if os.path.exists(Q_FILE):
#         try:
#             with open(Q_FILE, "r") as f:
#                 Q = json.load(f)
#             if not isinstance(Q, dict):
#                 Q = {}
#             print(f"✅ Loaded Q-table with {len(Q)} states")
#             return Q
#         except Exception as e:
#             print(f"⚠️ Q-table corrupted ({e}). Resetting...")
#             return {}
#     return {}

# def save_q_table(Q):
#     """Save Q-table to file"""
#     try:
#         with open(Q_FILE, "w") as f:
#             json.dump(Q, f, indent=4)
#         return True
#     except Exception as e:
#         print(f"❌ Failed to save Q-table: {e}")
#         return False

# Q = load_q_table()

# def get_epsilon():
#     """Get current epsilon value with decay"""
#     global EPISODE_COUNT
#     epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** EPISODE_COUNT))
#     return epsilon

# def get_state(skills, gaps, subject=""):
#     """Enhanced state representation"""
#     # Add subject info to state for better differentiation
#     skill_count = min(len(skills), 10)  # Cap at 10
#     gap_count = min(len(gaps), 10)       # Cap at 10
#     return f"{subject[:10]}_{skill_count}_{gap_count}"

# def choose_action(state, ml_suggestion, Q):
#     """Choose action using epsilon-greedy with ML bonus"""
#     if state not in Q:
#         Q[state] = {a: 0.0 for a in actions}
    
#     epsilon = get_epsilon()
    
#     # Exploration
#     if random.random() < epsilon:
#         return random.choice(actions)
    
#     # Exploitation with ML bonus
#     scores = {}
#     for a in actions:
#         rl_score = Q[state][a]
#         ml_bonus = 2.0 if a == ml_suggestion else 0.0  # Increased ML influence
#         scores[a] = rl_score + ml_bonus
    
#     return max(scores, key=scores.get)

# def update_q(state, action, reward, Q):
#     """Update Q-value with improved reward shaping"""
#     if state not in Q:
#         Q[state] = {a: 0.0 for a in actions}
    
#     old_value = Q[state][action]
#     Q[state][action] += ALPHA * (reward - old_value)
#     save_q_table(Q)
#     return Q[state][action]

# # =========================
# # GRAPH OPERATIONS
# # =========================
# def build_graph(subject):
#     """Build prerequisite dependency graph"""
#     df = skills_df[skills_df["subject"] == subject]
#     if df.empty:
#         return {}
    
#     graph = {}
#     all_skills = set(df["skill"].unique())
    
#     for _, row in df.iterrows():
#         skill = row["skill"]
#         prereq = row["prerequisite"]
        
#         if prereq in ["none", "", "nan"] or prereq not in all_skills:
#             continue
        
#         # Build forward graph (prerequisite -> skill)
#         graph.setdefault(prereq, []).append(skill)
    
#     # Add skills with no prerequisites as starting points
#     for skill in all_skills:
#         if skill not in graph and not any(skill in deps for deps in graph.values()):
#             graph[skill] = []
    
#     return graph

# def get_paths(graph, start, path=None, visited=None):
#     """Get all possible paths from start node"""
#     if path is None:
#         path = [start]
#     if visited is None:
#         visited = set()
    
#     if start in visited:
#         return [path]
    
#     visited.add(start)
    
#     if start not in graph or not graph[start]:
#         return [path]
    
#     all_paths = []
#     for nxt in graph[start]:
#         if nxt not in path:
#             new_paths = get_paths(graph, nxt, path + [nxt], visited.copy())
#             all_paths.extend(new_paths)
    
#     return all_paths if all_paths else [path]

# def get_roadmaps(graph, start):
#     """Generate three roadmap types from graph"""
#     paths = get_paths(graph, start)
    
#     if not paths:
#         return {a: [start] for a in actions}
    
#     # Remove duplicates and sort by length
#     unique_paths = []
#     for p in paths:
#         if p not in unique_paths:
#             unique_paths.append(p)
    
#     unique_paths.sort(key=len)
    
#     roadmaps = {
#         "Beginner Path": unique_paths[-1] if unique_paths else [start],  # Longest
#         "Fast Track": unique_paths[0] if unique_paths else [start],      # Shortest
#         "Project-Based": [start]                                         # Default
#     }
    
#     # Try to find project-based path
#     for path in unique_paths:
#         if any('project' in skill.lower() or 'build' in skill.lower() for skill in path):
#             roadmaps["Project-Based"] = path
#             break
    
#     return roadmaps

# # =========================
# # ML MODEL TRAINING
# # =========================
# def train_ml_model():
#     """Train and save ML model"""
#     global vectorizer, model
    
#     print("🔄 Training ML model...")
#     vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
#     X = vectorizer.fit_transform(train_df["skills"])
#     y = train_df["preferred_roadmap"]
    
#     model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X, y)
    
#     # Save model
#     joblib.dump(model, MODEL_FILE)
#     joblib.dump(vectorizer, VECTORIZER_FILE)
#     print("✅ ML model trained and saved")
#     return model, vectorizer

# # Load or train model
# if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
#     try:
#         model = joblib.load(MODEL_FILE)
#         vectorizer = joblib.load(VECTORIZER_FILE)
#         print("✅ Loaded pre-trained ML model")
#     except Exception as e:
#         print(f"⚠️ Could not load model ({e}), training new one...")
#         model, vectorizer = train_ml_model()
# else:
#     model, vectorizer = train_ml_model()

# def predict_roadmap(skills):
#     """Predict preferred roadmap using ML model"""
#     if not skills:
#         return random.choice(actions)
    
#     try:
#         skills_text = " ".join(skills)
#         X_pred = vectorizer.transform([skills_text])
#         return model.predict(X_pred)[0]
#     except Exception as e:
#         print(f"⚠️ Prediction error: {e}")
#         return "Beginner Path"

# # =========================
# # SKILL ANALYSIS
# # =========================
# def get_required_skills(subject):
#     """Get all required skills for a subject"""
#     return set(skills_df[skills_df["subject"] == subject]["skill"].unique())

# def get_skill_gaps(skills, subject):
#     """Identify missing skills"""
#     required = get_required_skills(subject)
#     return list(required - set(skills))

# def calculate_skill_coverage(skills, subject):
#     """Calculate percentage of skills covered"""
#     required = get_required_skills(subject)
#     if not required:
#         return 0.0
#     coverage = len(set(skills) & required) / len(required)
#     return coverage

# def skill_based_recommendation(skills, subject):
#     """Recommend track based on skill coverage"""
#     coverage = calculate_skill_coverage(skills, subject)
    
#     if coverage < 0.3:
#         return "Beginner Path"
#     elif coverage < 0.7:
#         return "Fast Track"
#     else:
#         return "Project-Based"

# # =========================
# # INTENT DETECTION
# # =========================
# def wants_help(text):
#     """Detect if user wants help/guidance"""
#     help_keywords = ["help", "recommend", "best", "choose", "suggest", 
#                      "advice", "guide", "which", "what should"]
#     return any(keyword in text.lower() for keyword in help_keywords)

# def wants_cv_upload(text):
#     """Detect if user wants to upload CV"""
#     cv_keywords = ["cv", "resume", "upload", "my skills"]
#     return any(keyword in text.lower() for keyword in cv_keywords)

# # =========================
# # SKILL EXTRACTION
# # =========================
# career_aliases = {
#     "ai": "ai engineer",
#     "machine learning": "ai engineer",
#     "ml": "ai engineer",
#     "frontend": "frontend developer",
#     "front-end": "frontend developer",
#     "web dev": "frontend developer",
#     "data science": "data scientist",
#     "data scientist": "data scientist",
#     "backend": "backend developer",
#     "back-end": "backend developer",
#     "fullstack": "fullstack developer",
#     "full-stack": "fullstack developer"
# }

# def extract_career_and_skills(text):
#     """Extract career path and skills from text"""
#     text = normalize(text)
    
#     # Extract skills
#     skills = []
#     skill_set = skills_df["skill"].unique()
#     for skill in skill_set:
#         # Use word boundaries for better matching
#         if re.search(r'\b' + re.escape(skill) + r'\b', text):
#             skills.append(skill)
    
#     # Extract career
#     career = None
#     career_set = skills_df["subject"].unique()
#     for c in career_set:
#         if re.search(r'\b' + re.escape(c) + r'\b', text):
#             career = c
#             break
    
#     # Check aliases
#     if not career:
#         for alias, mapped_career in career_aliases.items():
#             if alias in text:
#                 career = mapped_career
#                 break
    
#     return career, list(set(skills))

# def extract_skills_from_cv(cv_text):
#     """Extract skills from CV text with improved matching"""
#     cv_text = normalize(cv_text)
#     words = set(cv_text.split())
    
#     extracted = []
#     skill_set = skills_df["skill"].unique()
    
#     for skill in skill_set:
#         # Check for exact word match or phrase match
#         if skill in words:
#             extracted.append(skill)
#         elif ' ' in skill and skill in cv_text:
#             extracted.append(skill)
    
#     return list(set(extracted))

# # =========================
# # USER INPUT HANDLING
# # =========================
# def get_user_input():
#     """Get and parse user input"""
#     print("\n" + "="*50)
#     print("💬 How can I help you today?")
#     print("   (Type 'cv' to paste your CV, or describe your situation)")
#     print("="*50)
    
#     user_text = input("\nYou: ").strip()
    
#     if not user_text:
#         print("❌ Please enter some input")
#         return get_user_input()
    
#     # Handle CV upload
#     if user_text.lower() == 'cv':
#         print("\n📄 Please paste your CV content (press Enter twice to finish):")
#         lines = []
#         while True:
#             line = input()
#             if line == "" and lines and lines[-1] == "":
#                 break
#             lines.append(line)
#         cv_text = " ".join(lines)
        
#         skills = extract_skills_from_cv(cv_text)
#         print(f"\n✅ Extracted {len(skills)} skills from CV")
        
#         subject = input("\n🎯 What career path are you interested in? ").strip()
#         if subject:
#             subject = normalize(subject)
        
#         return subject, skills, True
    
#     # Parse normal input
#     subject, skills = extract_career_and_skills(user_text)
    
#     # Ask for missing information
#     if not subject:
#         subject = normalize(input("\n🎯 What career path are you interested in? "))
    
#     if not skills:
#         skills_input = input("🔧 What skills do you currently have? (comma-separated): ")
#         skills = [normalize(s.strip()) for s in skills_input.split(",") if s.strip()]
    
#     # Verify subject exists
#     if subject not in skills_df["subject"].unique():
#         print(f"⚠️ Career '{subject}' not found in database")
#         similar = [s for s in skills_df["subject"].unique() if subject in s or s in subject]
#         if similar:
#             print(f"💡 Did you mean: {', '.join(similar[:3])}?")
#             subject = similar[0] if len(similar) == 1 else subject
    
#     print(f"\n✅ Detected Career: {subject}")
#     print(f"✅ Detected Skills: {skills if skills else 'None'}")
    
#     return subject, skills, wants_help(user_text)

# # =========================
# # ROADMAP GENERATION
# # =========================
# def generate_roadmap(subject, skills, help_mode=False):
#     """Main roadmap generation logic"""
#     global EPISODE_COUNT
    
#     # Build graph
#     graph = build_graph(subject)
#     if not graph:
#         print(f"❌ No data found for career: {subject}")
#         return None
    
#     # Analyze skills
#     gaps = get_skill_gaps(skills, subject)
#     coverage = calculate_skill_coverage(skills, subject)
#     state = get_state(skills, gaps, subject)
    
#     # Find starting point
#     if skills:
#         start = next((s for s in skills if s in graph), list(graph.keys())[0])
#     else:
#         start = list(graph.keys())[0]
    
#     # Generate all roadmaps
#     roadmaps = get_roadmaps(graph, start)
    
#     # Get recommendations
#     skill_rec = skill_based_recommendation(skills, subject)
#     ml_rec = predict_roadmap(skills)
    
#     print(f"\n📊 Skill Coverage: {coverage:.1%}")
#     print(f"📊 Skill Gaps: {len(gaps)} missing skills")
    
#     # Handle help mode with RL
#     if help_mode and gaps:
#         EPISODE_COUNT += 1
        
#         print(f"\n🤖 ML Suggestion: {ml_rec}")
#         print(f"🎯 Skill-based Suggestion: {skill_rec}")
        
#         # Try RL-enhanced recommendation
#         best_action = choose_action(state, ml_rec, Q)
        
#         print(f"\n✨ AI Recommended Track: {best_action}")
#         print("\n📚 Learning Path:")
#         for i, step in enumerate(roadmaps[best_action], 1):
#             print(f"   {i}. {step}")
        
#         feedback = input("\n👍 Is this roadmap helpful? (y/n/m for more options): ").lower()
        
#         if feedback == 'y':
#             update_q(state, best_action, 1.0, Q)
#             return roadmaps[best_action], best_action, gaps
        
#         elif feedback == 'm':
#             print("\n🔄 Alternative options:")
#             for i, action in enumerate(actions, 1):
#                 if action != best_action:
#                     print(f"   {i}. {action}")
#                     for j, step in enumerate(roadmaps[action][:3], 1):
#                         print(f"      {j}. {step}")
#                     if len(roadmaps[action]) > 3:
#                         print(f"      ... and {len(roadmaps[action])-3} more steps")
            
#             choice = input("\nChoose option (1/2): ").strip()
#             if choice == '1':
#                 selected = [a for a in actions if a != best_action][0]
#                 update_q(state, selected, 0.5, Q)
#                 return roadmaps[selected], selected, gaps
#             elif choice == '2':
#                 selected = [a for a in actions if a != best_action][1]
#                 update_q(state, selected, 0.5, Q)
#                 return roadmaps[selected], selected, gaps
#             else:
#                 update_q(state, best_action, -0.5, Q)
#                 return roadmaps[skill_rec], skill_rec, gaps
#         else:
#             update_q(state, best_action, -0.5, Q)
#             return roadmaps[skill_rec], skill_rec, gaps
    
#     # Auto mode (no help needed or no gaps)
#     final_track = skill_rec if not help_mode else ml_rec
#     return roadmaps[final_track], final_track, gaps

# # =========================
# # DISPLAY RESULTS
# # =========================
# def display_roadmap(roadmap, track, gaps, subject):
#     """Display the final roadmap"""
#     if not roadmap:
#         return
    
#     print("\n" + "="*60)
#     print(f"🎯 YOUR PERSONALIZED {track.upper()} FOR {subject.upper()}")
#     print("="*60)
    
#     # Show skill gaps
#     if gaps:
#         print(f"\n📌 SKILLS TO LEARN ({len(gaps)}):")
#         for i, gap in enumerate(gaps[:10], 1):
#             print(f"   {i}. {gap}")
#         if len(gaps) > 10:
#             print(f"   ... and {len(gaps)-10} more")
#     else:
#         print("\n🎉 EXCELLENT! You have all the required skills!")
    
#     # Show roadmap
#     print(f"\n📚 LEARNING PATH ({len(roadmap)} steps):")
#     print("   " + "→".join(["📖"] * len(roadmap)))
    
#     for i, step in enumerate(roadmap, 1):
#         # Add emoji indicators
#         if i == 1:
#             prefix = "🚀"
#         elif i == len(roadmap):
#             prefix = "🏁"
#         else:
#             prefix = "📘"
#         print(f"   {prefix} Step {i}: {step}")
    
#     # Add estimated time
#     estimated_hours = len(roadmap) * 40  # Rough estimate: 40 hours per skill
#     weeks = estimated_hours // 40
#     print(f"\n⏱️ Estimated time: ~{weeks} weeks (studying 10 hours/week)")
    
#     # Add next step recommendation
#     if gaps:
#         print(f"\n💡 NEXT STEP: Start with '{gaps[0]}'")
    
#     print("\n" + "="*60)

# # =========================
# # MAIN LOOP
# # =========================
# def run():
#     """Main execution function"""
#     print("\n" + "="*60)
#     print("🤖 SMART MENTOR AI - Your Personalized Learning Assistant")
#     print("="*60)
    
#     # Get input
#     subject, skills, help_mode = get_user_input()
    
#     if not subject or subject not in skills_df["subject"].unique():
#         print(f"❌ Career '{subject}' not found in database")
#         print(f"Available careers: {', '.join(skills_df['subject'].unique()[:10])}")
#         return
    
#     # Generate roadmap
#     result = generate_roadmap(subject, skills, help_mode)
    
#     if result:
#         roadmap, track, gaps = result
#         display_roadmap(roadmap, track, gaps, subject)
        
#         # Save session summary
#         session_data = {
#             "subject": subject,
#             "skills": skills,
#             "track": track,
#             "gaps_count": len(gaps),
#             "steps": len(roadmap)
#         }
        
#         try:
#             with open("session_log.json", "a") as f:
#                 json.dump(session_data, f)
#                 f.write("\n")
#         except:
#             pass

# # =========================
# # PROGRAM ENTRY POINT
# # =========================
# if __name__ == "__main__":
#     print("\n🎓 WELCOME TO SMART MENTOR AI 🎓")
#     print("I'll help you create a personalized learning path!")
    
#     # Display available careers
#     careers = skills_df["subject"].unique()
#     print(f"\n📚 Available careers: {', '.join(careers[:15])}")
#     if len(careers) > 15:
#         print(f"   ... and {len(careers)-15} more")
    
#     # Main interaction loop
#     while True:
#         try:
#             run()
            
#             again = input("\n🔄 Would you like another recommendation? (y/n): ").strip().lower()
#             if again != 'y':
#                 print("\n👋 Thank you for using Smart Mentor AI! Good luck with your learning journey!")
#                 break
                
#         except KeyboardInterrupt:
#             print("\n\n👋 Goodbye! Happy learning!")
#             break
#         except Exception as e:
#             print(f"\n❌ An error occurred: {e}")
#             print("Please try again or restart the application")
            
#             if input("\nContinue? (y/n): ").lower() != 'y':
#                 break


import pandas as pd
import re
import random
import json
import os
import pickle
import ast  # Add this for safe evaluation
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
import time
from datetime import datetime
from collections import defaultdict
import hashlib

# Modern ML imports
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

# =========================
# DATA MODELS
# =========================
@dataclass
class UserProfile:
    """Modern user profile with learning analytics"""
    user_id: str
    skills: List[str]
    career_goal: str
    learning_style: str = "balanced"
    pace: str = "moderate"
    available_hours_per_week: int = 10
    experience_level: str = "beginner"
    learning_history: List[Dict] = None
    created_at: str = None
    last_active: str = None
    
    def __post_init__(self):
        if self.learning_history is None:
            self.learning_history = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.last_active = datetime.now().isoformat()

@dataclass
class LearningPath:
    """Modern learning path structure"""
    title: str
    track_type: str
    skills: List[str]
    estimated_hours: int
    difficulty: str
    prerequisites: List[str]
    milestones: List[Dict]
    resources: List[Dict]
    projects: List[Dict]
    quizzes: List[Dict]
    created_at: str

@dataclass
class SkillNode:
    """Graph node with embeddings"""
    name: str
    prerequisites: List[str]
    difficulty: int
    estimated_hours: int
    embedding: np.ndarray = None
    resources: List[str] = None

# =========================
# MODERN AI MODELS
# =========================
class ModernSkillExtractor:
    """Uses transformer models for intelligent skill extraction"""
    
    def __init__(self):
        console.print("[bold cyan]🚀 Loading AI models...[/bold cyan]")
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading BERT model...", total=None)
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load BERT model: {e}[/yellow]")
            console.print("[yellow]Using fallback skill extraction...[/yellow]")
            self.model = None
        
        self.skill_embeddings = {}
        self.skill_synonyms = self._load_synonyms()
    
    def _load_synonyms(self) -> Dict:
        """Load skill synonyms and related terms"""
        return {
            'python': ['py', 'python3', 'python programming'],
            'javascript': ['js', 'ecmascript', 'javascript programming'],
            'machine learning': ['ml', 'machine learning ai', 'predictive modeling'],
            'deep learning': ['dl', 'neural networks', 'deep neural networks'],
            'sql': ['structured query language', 'postgresql', 'mysql'],
            'react': ['reactjs', 'react.js', 'react framework'],
            'aws': ['amazon web services', 'ec2', 's3'],
            'docker': ['containerization', 'docker containers', 'docker engine'],
            'java': ['java8', 'java11', 'core java'],
            'c++': ['cpp', 'c plus plus'],
            'html': ['html5', 'html/css'],
            'css': ['css3', 'stylesheet'],
        }
    
    def extract_skills(self, text: str, all_skills: List[str]) -> List[Tuple[str, float]]:
        """Extract skills with confidence scores"""
        text_lower = text.lower()
        detected = []
        
        # Direct matching with synonyms
        for skill in all_skills:
            confidence = 0.0
            skill_lower = skill.lower()
            
            if skill_lower in text_lower:
                confidence = 0.95
            else:
                # Check synonyms
                for key, synonyms in self.skill_synonyms.items():
                    if key in skill_lower:
                        for syn in synonyms:
                            if syn in text_lower:
                                confidence = 0.85
                                break
                    if confidence > 0:
                        break
            
            if confidence > 0:
                detected.append((skill, confidence))
        
        # Use embeddings for semantic matching (if available)
        if self.model and len(detected) < 5 and len(text) > 50:
            try:
                text_embedding = self.model.encode([text])[0]
                
                for skill in all_skills:
                    if skill not in [d[0] for d in detected]:
                        if skill not in self.skill_embeddings:
                            self.skill_embeddings[skill] = self.model.encode([skill])[0]
                        
                        similarity = cosine_similarity([text_embedding], [self.skill_embeddings[skill]])[0][0]
                        if similarity > 0.6:
                            detected.append((skill, similarity))
            except:
                pass
        
        return sorted(detected, key=lambda x: x[1], reverse=True)[:15]

class Neo4jStyleGraph:
    """Graph database with vector similarity search"""
    
    def __init__(self):
        self.nodes: Dict[str, SkillNode] = {}
        self.adjacency = defaultdict(list)
    
    def add_node(self, node: SkillNode):
        self.nodes[node.name] = node
    
    def add_edge(self, from_skill: str, to_skill: str):
        if from_skill in self.nodes and to_skill in self.nodes:
            self.adjacency[from_skill].append(to_skill)
    
    def find_path(self, start: str, end: str = None) -> List[List[str]]:
        """Find all paths using BFS"""
        paths = []
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            node, path = queue.pop(0)
            
            if node in visited and len(path) > 10:  # Prevent infinite loops
                continue
            
            visited.add(node)
            
            if node in self.adjacency:
                for neighbor in self.adjacency[node]:
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        paths.append(new_path)
                        queue.append((neighbor, new_path))
        
        return paths
    
    def get_optimal_path(self, start: str, user_profile: UserProfile) -> Dict:
        """Get personalized optimal path based on user profile"""
        all_paths = self.find_path(start)
        
        if not all_paths:
            return {"Beginner Path": [start]}
        
        # Score each path based on user preferences
        scored_paths = []
        for path in all_paths:
            score = 0
            # Longer paths for detailed learners
            if user_profile.learning_style in ["visual", "reading"]:
                score += len(path) * 0.5
            else:
                score -= len(path) * 0.3
            
            # Faster paths for experienced users
            if user_profile.experience_level == "advanced":
                score += (1 / max(len(path), 1)) * 10
            
            scored_paths.append((score, path))
        
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        # Ensure we have unique paths
        unique_paths = []
        for _, path in scored_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        if len(unique_paths) >= 3:
            return {
                "Beginner Path": max(unique_paths, key=len),
                "Fast Track": min(unique_paths, key=len),
                "Recommended": unique_paths[0]
            }
        else:
            return {
                "Beginner Path": unique_paths[0] if unique_paths else [start],
                "Fast Track": unique_paths[0] if unique_paths else [start],
                "Recommended": unique_paths[0] if unique_paths else [start]
            }

# =========================
# MODERN RL AGENT
# =========================
class ModernQLearningAgent:
    """Deep Q-Learning with experience replay"""
    
    def __init__(self, actions: List[str], learning_rate=0.1, discount_factor=0.95):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = []
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        self.min_exploration = 0.05
        
    def get_state(self, user_profile: UserProfile, gaps: List[str]) -> str:
        """Create rich state representation"""
        return f"{user_profile.career_goal}|{len(user_profile.skills)}|{len(gaps)}|{user_profile.experience_level}|{user_profile.learning_style}"
    
    def choose_action(self, state: str, ml_suggestion: str) -> str:
        """Epsilon-greedy action selection with ML guidance"""
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        
        # Get Q-values with ML bonus
        q_values = {}
        for action in self.actions:
            q_value = self.q_table[state][action]
            ml_bonus = 1.5 if action == ml_suggestion else 0
            q_values[action] = q_value + ml_bonus
        
        return max(q_values, key=q_values.get)
    
    def update(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value with experience replay"""
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
        
        # Update Q-value
        best_next = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        
        # Decay exploration
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
    
    def save(self, path: str):
        """Save Q-table to file"""
        try:
            with open(path, 'w') as f:
                json.dump({k: dict(v) for k, v in self.q_table.items()}, f)
        except:
            pass
    
    def load(self, path: str):
        """Load Q-table from file"""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.q_table = defaultdict(lambda: defaultdict(float), {
                        k: defaultdict(float, v) for k, v in data.items()
                    })
            except:
                pass

# =========================
# MODERN UI COMPONENTS
# =========================
class ModernUI:
    """Beautiful terminal UI with rich formatting"""
    
    @staticmethod
    def show_banner():
        """Display animated banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗                ║
║   ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝                ║
║   ███████╗██╔████╔██║███████║██████╔╝   ██║                   ║
║   ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║                   ║
║   ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║                   ║
║   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                   ║
║                                                               ║
║                    SMART MENTOR AI v3.0                       ║
║              Your AI-Powered Learning Companion               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(Text(banner, style="bold cyan"), border_style="cyan"))
    
    @staticmethod
    def show_analytics_dashboard(user_profile: UserProfile, learning_path: LearningPath):
        """Display modern analytics dashboard"""
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header with user info
        header_text = Text()
        header_text.append(f"👤 {user_profile.user_id} | ", style="bold white")
        header_text.append(f"🎯 {user_profile.career_goal} | ", style="bold green")
        header_text.append(f"⚡ {user_profile.pace.upper()} | ", style="bold yellow")
        header_text.append(f"📚 {user_profile.experience_level.upper()}", style="bold blue")
        
        layout["header"].update(Panel(header_text, title="User Dashboard", border_style="cyan"))
        
        # Body with metrics
        metrics_table = Table(title="📊 Learning Analytics", style="cyan")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Progress", style="green")
        
        progress_percent = min(100, int((len(user_profile.skills) / max(len(learning_path.skills), 1)) * 100))
        progress_bar = "█" * (progress_percent // 10) + "░" * (10 - (progress_percent // 10))
        
        metrics_table.add_row(
            "Skills Mastered",
            str(len(user_profile.skills)),
            f"{progress_bar} {progress_percent}%"
        )
        metrics_table.add_row(
            "Learning Path Progress",
            f"0/{len(learning_path.skills)}",
            "░░░░░░░░░░ 0%"
        )
        metrics_table.add_row(
            "Estimated Completion",
            f"{learning_path.estimated_hours} hours",
            f"{learning_path.estimated_hours // max(user_profile.available_hours_per_week, 1)} weeks"
        )
        metrics_table.add_row(
            "Learning Style",
            user_profile.learning_style.title(),
            "Optimized ✓"
        )
        
        layout["body"].update(metrics_table)
        
        # Footer with motivational quote
        quotes = [
            "💡 The expert in anything was once a beginner",
            "🚀 Consistency beats intensity",
            "🎯 Every expert was once a beginner",
            "📚 Knowledge is power",
            "🌟 The future belongs to those who learn"
        ]
        layout["footer"].update(Panel(random.choice(quotes), style="italic green"))
        
        console.print(layout)
    
    @staticmethod
    def show_learning_path(learning_path: LearningPath):
        """Display interactive learning path"""
        console.print("\n")
        console.print(Panel(f"[bold cyan]📚 {learning_path.title}[/bold cyan]", border_style="cyan"))
        
        # Timeline view
        for i, skill in enumerate(learning_path.skills[:10], 1):  # Show first 10 skills
            # Create progress bar for each skill
            progress_percent = int((i / max(len(learning_path.skills), 1)) * 100)
            progress_bar = "█" * (progress_percent // 10) + "░" * (10 - (progress_percent // 10))
            
            skill_panel = Panel(
                f"[bold yellow]Step {i}[/bold yellow]: [white]{skill}[/white]\n"
                f"[dim]├─ Difficulty: {random.choice(['⭐', '⭐⭐', '⭐⭐⭐'])}[/dim]\n"
                f"[dim]├─ Est. Time: {random.randint(20, 60)} hours[/dim]\n"
                f"[dim]└─ Progress: {progress_bar} {progress_percent}%[/dim]",
                border_style="blue"
            )
            console.print(skill_panel)
        
        if len(learning_path.skills) > 10:
            console.print(f"\n[dim]... and {len(learning_path.skills) - 10} more skills[/dim]")
        
        # Show projects
        if learning_path.projects:
            console.print("\n[bold magenta]🎯 Practical Projects:[/bold magenta]")
            for project in learning_path.projects[:3]:
                console.print(f"  • {project['name']} - {project['description']}")
        
        # Show resources
        if learning_path.resources:
            console.print("\n[bold green]📖 Recommended Resources:[/bold green]")
            for resource in learning_path.resources[:3]:
                console.print(f"  • {resource['type']}: {resource['title']}")

# =========================
# MAIN APPLICATION
# =========================
class SmartMentorAI:
    """Main application class"""
    
    def __init__(self):
        self.ui = ModernUI()
        self.skill_extractor = None
        self.graph = None
        self.rl_agent = None
        self.current_user: Optional[UserProfile] = None
        self.all_skills = []
        self.careers = []
        
        self._initialize()
    
    def _safe_parse_skills(self, skills_str):
        """Safely parse skills from string representation"""
        if pd.isna(skills_str):
            return []
        
        try:
            # Try to evaluate as Python literal
            if isinstance(skills_str, str):
                # Remove any quotes and brackets
                cleaned = skills_str.strip()
                if cleaned.startswith('[') and cleaned.endswith(']'):
                    # It's a list representation
                    try:
                        return ast.literal_eval(cleaned)
                    except:
                        # Fallback: split by comma and clean
                        return [s.strip().strip("'\"") for s in cleaned[1:-1].split(',')]
                else:
                    # Single skill or comma-separated
                    return [s.strip().strip("'\"") for s in cleaned.split(',')]
            return []
        except:
            return []
    
    def _initialize(self):
        """Initialize all components"""
        self.ui.show_banner()
        
        # Load data
        with console.status("[bold green]Loading knowledge base...") as status:
            try:
                # Load CSV files with error handling
                skills_path = r"D:/Games/RoadMap Ai/skills_roadmap.csv"
                train_path = r"D:/Games/RoadMap Ai/training_data22.csv"
                
                if not os.path.exists(skills_path):
                    console.print(f"[red]❌ File not found: {skills_path}[/red]")
                    raise FileNotFoundError(f"Skills roadmap file not found")
                
                if not os.path.exists(train_path):
                    console.print(f"[red]❌ File not found: {train_path}[/red]")
                    raise FileNotFoundError(f"Training data file not found")
                
                self.skills_df = pd.read_csv(skills_path)
                self.train_df = pd.read_csv(train_path)
                
                # Normalize data
                for col in ['skill', 'prerequisite', 'subject']:
                    if col in self.skills_df.columns:
                        self.skills_df[col] = self.skills_df[col].astype(str).str.lower().str.strip()
                
                self.all_skills = self.skills_df['skill'].unique().tolist()
                self.careers = self.skills_df['subject'].unique().tolist()
                
                console.print(f"[green]✓ Loaded {len(self.all_skills)} skills and {len(self.careers)} careers[/green]")
                
                # Initialize AI components
                self.skill_extractor = ModernSkillExtractor()
                self._build_knowledge_graph()
                self._init_ml_model()
                
                # Initialize RL agent
                self.rl_agent = ModernQLearningAgent(
                    actions=["Beginner Path", "Fast Track", "Project-Based", "Recommended"]
                )
                self.rl_agent.load("q_table_modern.json")
                
                status.update("[bold green]✓ Ready![/bold green]")
                time.sleep(1)
            except Exception as e:
                console.print(f"[red]❌ Initialization failed: {e}[/red]")
                console.print("[yellow]Please check your data files and try again[/yellow]")
                raise
    
    def _build_knowledge_graph(self):
        """Build knowledge graph with embeddings"""
        self.graph = Neo4jStyleGraph()
        
        # Add nodes
        for skill in self.all_skills[:100]:  # Limit for performance
            node = SkillNode(
                name=skill,
                prerequisites=[],
                difficulty=random.randint(1, 5),
                estimated_hours=random.randint(20, 80),
                resources=[]
            )
            self.graph.add_node(node)
        
        # Add edges
        for _, row in self.skills_df.iterrows():
            prereq = row.get('prerequisite', 'none')
            skill = row.get('skill', '')
            if prereq not in ['none', '', 'nan'] and skill in self.graph.nodes and prereq in self.graph.nodes:
                self.graph.add_edge(prereq, skill)
    
    def _init_ml_model(self):
        """Initialize ML model for path prediction"""
        try:
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.label_encoder = LabelEncoder()
            
            # Safely parse skills column
            if 'skills' in self.train_df.columns and 'preferred_roadmap' in self.train_df.columns:
                # Parse skills safely
                parsed_skills = []
                for skills_str in self.train_df['skills']:
                    skills_list = self._safe_parse_skills(skills_str)
                    parsed_skills.append(' '.join(skills_list))
                
                # Use length as simple feature
                X = [[len(skills.split())] for skills in parsed_skills]
                y = self.label_encoder.fit_transform(self.train_df['preferred_roadmap'].fillna('Beginner Path'))
                
                # Train if we have data
                if len(X) > 0:
                    self.ml_model.fit(X, y)
                    console.print("[green]✓ ML model trained successfully[/green]")
                else:
                    console.print("[yellow]⚠️ No training data available[/yellow]")
            else:
                console.print("[yellow]⚠️ Training data missing required columns[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️ ML model initialization failed: {e}[/yellow]")
    
    def create_user_profile(self) -> UserProfile:
        """Interactive user profile creation"""
        console.print("\n[bold cyan]✨ Let's create your learning profile[/bold cyan]\n")
        
        name = Prompt.ask("[yellow]What's your name?[/yellow]")
        
        # Learning style assessment
        console.print("\n[cyan]📚 How do you learn best?[/cyan]")
        learning_style = Prompt.ask(
            "Choose your style",
            choices=["visual", "reading", "hands-on", "balanced"],
            default="balanced"
        )
        
        # Experience level
        console.print("\n[cyan]⭐ What's your experience level?[/cyan]")
        experience = Prompt.ask(
            "Select your level",
            choices=["beginner", "intermediate", "advanced"],
            default="beginner"
        )
        
        # Available time
        hours = Prompt.ask(
            "[cyan]⏰ How many hours can you study per week?[/cyan]",
            default="10"
        )
        
        # Career goal
        console.print(f"\n[cyan]🎯 Available careers: {', '.join(self.careers[:10])}[/cyan]")
        if len(self.careers) > 10:
            console.print(f"[dim]... and {len(self.careers)-10} more[/dim]")
        
        career = Prompt.ask("[yellow]What's your career goal?[/yellow]")
        
        # Initial skills
        console.print("\n[cyan]🔧 Tell me about your current skills[/cyan]")
        console.print("[dim]Example: Python, JavaScript, SQL, or describe your experience[/dim]")
        skills_text = Prompt.ask("Your skills")
        
        # Extract skills using AI
        extracted = self.skill_extractor.extract_skills(skills_text, self.all_skills)
        skills = [skill for skill, conf in extracted if conf > 0.6]
        
        if not skills:
            skills = [s.strip().lower() for s in skills_text.split(',') if s.strip()]
        
        console.print(f"\n[green]✓ Extracted {len(skills)} skills from your input[/green]")
        if skills:
            console.print(f"[dim]Skills: {', '.join(skills[:5])}[/dim]")
        
        return UserProfile(
            user_id=hashlib.md5(name.encode()).hexdigest()[:8],
            skills=skills,
            career_goal=career,
            learning_style=learning_style,
            pace="moderate",
            available_hours_per_week=int(hours),
            experience_level=experience
        )
    
    def generate_personalized_path(self, user_profile: UserProfile) -> Tuple[LearningPath, str]:
        """Generate personalized learning path"""
        console.print("\n[bold cyan]🧠 Generating your personalized learning path...[/bold cyan]")
        
        with Progress() as progress:
            task = progress.add_task("Analyzing skill gaps...", total=4)
            
            # Find skill gaps
            career_skills = self.skills_df[self.skills_df['subject'] == user_profile.career_goal]
            required_skills = set(career_skills['skill']) if not career_skills.empty else set()
            current_skills = set(user_profile.skills)
            gaps = list(required_skills - current_skills)
            progress.advance(task)
            
            # Find starting point
            if current_skills:
                start_skill = next((s for s in current_skills if s in self.graph.nodes), gaps[0] if gaps else None)
            else:
                start_skill = gaps[0] if gaps else None
            
            if not start_skill and self.graph.nodes:
                start_skill = list(self.graph.nodes.keys())[0]
            
            if not start_skill:
                start_skill = user_profile.career_goal
            
            progress.advance(task)
            
            # Get path from graph
            if self.graph and start_skill in self.graph.nodes:
                paths = self.graph.get_optimal_path(start_skill, user_profile)
            else:
                paths = {"Beginner Path": [start_skill], "Fast Track": [start_skill], "Recommended": [start_skill]}
            
            # Select best path using RL
            state = self.rl_agent.get_state(user_profile, gaps)
            ml_suggestion = "Recommended"
            selected_track = self.rl_agent.choose_action(state, ml_suggestion)
            
            path_skills = paths.get(selected_track, paths["Beginner Path"])
            progress.advance(task)
            
            # Create learning path object
            learning_path = LearningPath(
                title=f"Master {user_profile.career_goal.title()} - {selected_track}",
                track_type=selected_track,
                skills=path_skills[:20],  # Limit to 20 skills
                estimated_hours=len(path_skills) * 40,
                difficulty="intermediate",
                prerequisites=[s for s in path_skills[:3] if s in current_skills],
                milestones=[
                    {"title": f"Complete {skill}", "hours": 40} for skill in path_skills[:5]
                ],
                resources=[
                    {"type": "Course", "title": f"Complete {skill} Bootcamp", "url": "#"}
                    for skill in path_skills[:3]
                ],
                projects=[
                    {"name": f"Build {skill} Project", "description": f"Hands-on project for {skill}"}
                    for skill in path_skills[:2]
                ],
                quizzes=[
                    {"title": f"{skill} Assessment", "questions": 20}
                    for skill in path_skills[:2]
                ],
                created_at=datetime.now().isoformat()
            )
            progress.advance(task)
        
        return learning_path, selected_track
    
    def get_user_feedback(self, track: str) -> float:
        """Collect and process user feedback"""
        console.print("\n[cyan]💬 How was this recommendation?[/cyan]")
        feedback = Prompt.ask(
            "Rate this path",
            choices=["excellent", "good", "okay", "poor"],
            default="good"
        )
        
        reward_map = {"excellent": 1.0, "good": 0.5, "okay": 0.0, "poor": -0.5}
        return reward_map.get(feedback, 0)
    
    def run(self):
        """Main application loop"""
        try:
            # Create user profile
            self.current_user = self.create_user_profile()
            
            while True:
                # Generate learning path
                learning_path, track = self.generate_personalized_path(self.current_user)
                
                # Show dashboard
                self.ui.show_analytics_dashboard(self.current_user, learning_path)
                
                # Show learning path
                self.ui.show_learning_path(learning_path)
                
                # Get feedback for RL
                reward = self.get_user_feedback(track)
                
                # Update RL agent
                state = self.rl_agent.get_state(self.current_user, [])
                self.rl_agent.update(state, track, reward, state)
                self.rl_agent.save("q_table_modern.json")
                
                # Ask for continuation
                if not Confirm.ask("\n[cyan]🔄 Would you like to explore another path?[/cyan]", default=False):
                    console.print("\n[bold green]🎉 Happy learning! Remember: consistency is key![/bold green]")
                    break
                
                # Update user skills (simulate learning)
                if Confirm.ask("Have you learned any new skills?", default=False):
                    new_skills = Prompt.ask("List new skills (comma-separated)").split(',')
                    self.current_user.skills.extend([s.strip().lower() for s in new_skills if s.strip()])
                    self.current_user.skills = list(set(self.current_user.skills))
                    console.print(f"[green]✓ Updated! You now have {len(self.current_user.skills)} skills[/green]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye! Keep learning![/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            import traceback
            traceback.print_exc()

# =========================
# RUN APPLICATION
# =========================
if __name__ == "__main__":
    # Check and install required packages
    required_packages = ['rich', 'pandas', 'scikit-learn', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        console.print(f"[yellow]📦 Installing required packages: {', '.join(missing)}[/yellow]")
        for package in missing:
            os.system(f"pip install {package}")
    
    # Try to install sentence-transformers (optional)
    try:
        import sentence_transformers
    except ImportError:
        console.print("[yellow]⚠️ sentence-transformers not installed. Using fallback skill extraction.[/yellow]")
        console.print("[dim]For better results, run: pip install sentence-transformers[/dim]")
    
    # Run the modern mentor
    try:
        mentor = SmartMentorAI()
        mentor.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        console.print("[yellow]Please ensure your CSV files exist at the correct paths[/yellow]")
        input("Press Enter to exit...")