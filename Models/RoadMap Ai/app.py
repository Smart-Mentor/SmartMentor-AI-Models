"""
Career Roadmap AI — FastAPI for Hugging Face Spaces
All heavy LLMs removed. Logic is identical to RoadMapAI.py.
All ML models removed. Pure rule-based pattern matching for skill detection.
Pure pattern matching — no ML models required.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import re
import os
import random
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from enum import Enum
import numpy as np

# No heavy ML models — pure pattern matching only

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SKILLS_PATH = os.getenv("SKILLS_PATH", "SkillsRoadmap.csv")
TRAIN_PATH  = os.getenv("TRAIN_PATH",  "TrainingData.csv")

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Career Roadmap AI",
    description="AI-powered career roadmap generator. Tell it your goal and skills, get a personalized learning path.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA CLASSES  (identical to original)
# =============================================================================
@dataclass
class Skill:
    name: str
    level: str
    duration_weeks: int
    prerequisites: List[str]
    category: str
    importance: float
    real_world_applications: List[str]

@dataclass
class CareerPath:
    name: str
    description: str
    required_skills: List[Skill]
    market_demand: str
    average_salary_range: str
    industry_sectors: List[str]

class ExperienceLevel(Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"

class UserInteractionState(Enum):
    INITIAL            = "initial"
    CAREER_DETECTED    = "career_detected"
    ASKING_CAREER      = "asking_career"      # skills known, waiting for career
    ASKING_SKILLS      = "asking_skills"
    SKILLS_PROVIDED    = "skills_provided"
    READY_TO_GENERATE  = "ready_to_generate"

# =============================================================================
# MODEL MANAGER  (stub only — all inference is rule-based)
# =============================================================================
class ModelManager:
    """
    All heavy models removed. Skill extraction is pure pattern matching.
    This class kept only so RoadmapGenerator API stays unchanged.
    """
    def __init__(self):
        print("✅ ModelManager ready (rule-based, no ML models)")

    def intent_classifier(self, text, labels, multi_label=False):
        return {"labels": labels, "scores": [0.0] * len(labels)}

    def text_generator(self, prompt, **kwargs):
        return None

# =============================================================================
# SKILL EXTRACTOR  (logic identical, heavy NER models removed)
# =============================================================================
class SkillExtractor:
    """
    Strict skill extractor — only returns skills that EXACTLY appear in the text.
    No semantic guessing. No free-text chunks. No false positives.

    Two strategies only:
      1. Exact word-boundary match against known_skills list  (primary)
      2. Comma/and-separated parts — each part validated against known_skills (secondary)

    Semantic similarity removed entirely — it was the main source of false positives.
    """

    # Words that look like skill words but are not skills
    NOISE_WORDS = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'by','from','up','about','into','through','during','before','after',
        'above','below','between','out','off','over','under','again','further',
        'then','once','here','there','when','where','why','how','all','both',
        'each','few','more','most','other','some','such','no','nor','not',
        'only','own','same','so','than','too','very','can','will','just',
        'should','now','also','been','being','have','has','had','does','did',
        'doing','would','could','this','that','these','i','me','my','myself',
        'we','our','you','your','he','she','it','they','them','know','want',
        'become','experience','knowledge','skill','skills','learn','learning',
        'study','studying','interested','interest','using','building','working',
        'creating','developing','familiar','understanding','basic','advanced',
        'none','no','nothing','na','nil','null','zero','scratch',
    }

    # Career-role words that must never be returned as skills
    CAREER_WORDS = {
        'frontend','backend','fullstack','full stack','full-stack',
        'frontend developer','backend developer','fullstack developer',
        'data scientist','data science','ai engineer','ml engineer',
        'cybersecurity','devops','mobile developer','game developer',
        'data engineer','software engineer','web developer','developer',
        'engineer','scientist',
    }

    def __init__(self, model_manager, known_skills: List[str]):
        # model_manager kept for API compatibility but no longer used here
        self.known_skills = set(s.lower().strip() for s in known_skills if s.strip())
        print(f"Skill Extractor ready ({len(self.known_skills)} known skills, strict mode)")

    # ── Public entry point ────────────────────────────────────────────────────
    def extract(self, text: str) -> List[str]:
        if not text or len(text.strip()) < 2:
            return []

        text = text.lower().strip()

        # Early exit: none/nothing/no skills indicators
        none_phrases = [
            'none','no skills','no experience','nothing','n/a','na','nil',
            'from scratch','complete beginner','total beginner','just starting',
            'no prior','brand new','starting fresh',
        ]
        if any(p in text for p in none_phrases):
            return []

        found = set()

        # Strategy 1: exact match against every known skill
        found.update(self._exact_match(text))

        # Strategy 2: split on delimiters, validate each part
        found.update(self._split_and_validate(text))

        # Remove career names and noise
        found = {s for s in found if s not in self.CAREER_WORDS}
        found = {s for s in found if s not in self.NOISE_WORDS}
        found = {s for s in found if len(s) > 1}

        return sorted(found)   # sorted for deterministic output

    # ── Strategy 1: exact word-boundary match ─────────────────────────────────
    def _exact_match(self, text: str) -> set:
        """
        Scans the full text for every known skill using:
          - word-boundary regex for plain-word skills  (python, sql, react …)
          - simple substring for skills with special chars  (c++, c#, .net …)
          - contiguous-phrase check for multi-word skills  (machine learning …)
        """
        matches = set()
        for skill in self.known_skills:
            if self._skill_in_text(skill, text):
                matches.add(skill)
        return matches

    def _skill_in_text(self, skill: str, text: str) -> bool:
        """Returns True only if `skill` genuinely appears in `text`."""
        try:
            if re.search(r'[^\w\s]', skill):
                # Special-char skill (c++, c#, .net, node.js) — literal match
                return bool(re.search(re.escape(skill), text))
            elif ' ' in skill:
                # Multi-word skill — must appear as a contiguous phrase
                return bool(re.search(r'\b' + re.escape(skill) + r'\b', text))
            else:
                # Single-word skill — strict word boundary
                return bool(re.search(r'\b' + re.escape(skill) + r'\b', text))
        except re.error:
            return skill in text

    # ── Strategy 2: delimiter-split → validate each chunk ─────────────────────
    def _split_and_validate(self, text: str) -> set:
        """
        Splits on , ; and with also → validates each chunk strictly.
        A chunk is accepted ONLY if it exactly equals a known skill
        (after stripping filler words). No free-text chunks ever added.
        """
        # Remove common filler before splitting
        cleaned = re.sub(
            r'\b(i|we|have|know|can|use|work with|worked with|experience with|'
            r'experienced in|proficient in|skilled in|familiar with|'
            r'some|basic|advanced|beginner|intermediate|expert|'
            r'a bit of|a little|lot of|years of|year of)\b',
            ' ', text, flags=re.IGNORECASE
        )
        parts = re.split(r'[,;]|\band\b|\bwith\b|\balso\b|\bplus\b', cleaned)

        found = set()
        for part in parts:
            part = part.strip().lower()
            if not part or part in self.NOISE_WORDS or len(part) < 2:
                continue
            # Only accept if this chunk IS a known skill
            if part in self.known_skills:
                found.add(part)
            # Or if a known skill appears inside this short chunk (≤4 words)
            elif len(part.split()) <= 4:
                for skill in self.known_skills:
                    if self._skill_in_text(skill, part):
                        found.add(skill)
        return found

# =============================================================================
# DIALOGUE MANAGER  (identical to original)
# =============================================================================
class DialogueManager:
    def __init__(self, available_careers: List[str]):
        self.state = UserInteractionState.INITIAL
        self.available_careers = available_careers
        self.detected_career = None
        self.detected_skills = []
        self.experience_level = "beginner"
        self.conversation_history = []

    def process_input(self, user_input: str, extracted_career: Optional[str],
                      extracted_skills: List[str]) -> Dict:
        user_input_lower = user_input.lower().strip()
        self.conversation_history.append({"role": "user", "content": user_input})

        if user_input_lower in ['quit', 'exit', 'q', 'stop']:
            return {"action": "exit"}
        if user_input_lower in ['restart', 'reset', 'start over']:
            self.reset()
            return {"action": "restart"}

        if self.state == UserInteractionState.INITIAL:
            return self._process_initial_input(extracted_career, extracted_skills)
        elif self.state == UserInteractionState.ASKING_CAREER:
            # Skills already stored — just need the career now
            return self._process_career_answer(user_input_lower, extracted_career)
        elif self.state == UserInteractionState.CAREER_DETECTED:
            return self._process_career_detected(user_input_lower, extracted_skills)
        elif self.state == UserInteractionState.ASKING_SKILLS:
            return self._process_skills_response(user_input_lower, extracted_skills)
        elif self.state == UserInteractionState.SKILLS_PROVIDED:
            return self._process_experience_response(user_input_lower)
        else:
            return {"action": "generate_roadmap"}

    def _process_initial_input(self, career: Optional[str], skills: List[str]) -> Dict:
        # Only update stored values when new info arrives — never wipe existing data
        if career:
            self.detected_career = career
        if skills:
            self.detected_skills = skills

        effective_career = self.detected_career
        effective_skills = self.detected_skills

        if effective_career and effective_skills:
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_career_and_skills",
                "career": effective_career,
                "skills": effective_skills,
                "message": f"Great! You want to become a {effective_career} and already know: {', '.join(effective_skills)}"
            }
        elif effective_career and not effective_skills:
            self.state = UserInteractionState.ASKING_SKILLS
            return {
                "action": "ask_for_skills",
                "career": effective_career,
                "message": self._generate_skills_prompt(effective_career)
            }
        elif not effective_career and effective_skills:
            # Skills collected — now ask for career and remember the state
            self.state = UserInteractionState.ASKING_CAREER
            careers_list = ", ".join(self.available_careers)
            return {
                "action": "ask_for_career",
                "skills": effective_skills,
                "message": (
                    f"Got it! I see you know: {', '.join(effective_skills)}\n"
                    f"What career are you aiming for?\n"
                    f"Available: {careers_list}"
                )
            }
        else:
            careers_list = ", ".join(self.available_careers)
            return {
                "action": "ask_everything",
                "message": (
                    "I couldn't recognise that input.\n\n"
                    "Please tell me:\n"
                    "1. Your target career — e.g. 'Data Scientist', 'Frontend Developer'\n"
                    f"   Available: {careers_list}\n\n"
                    "2. Skills you already know — e.g. 'Python, SQL'\n"
                    "   Or type 'none' to start from scratch."
                )
            }

    def _process_career_answer(self, user_input: str, extracted_career: Optional[str]) -> Dict:
        """Called when we already have skills and are waiting for the user to name a career."""
        # Try extracted career first, then fuzzy match against available careers
        career = extracted_career
        if not career:
            for c in self.available_careers:
                if c in user_input or any(w in user_input for w in c.split()):
                    career = c
                    break

        if career:
            self.detected_career = career
            # Skills were already stored — go straight to confirmation
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_career_and_skills",
                "career": career,
                "skills": self.detected_skills,
                "message": (
                    f"Got it! Career: {career}\n"
                    f"Your existing skills: {', '.join(self.detected_skills) if self.detected_skills else 'none'}\n"
                    "Generating your roadmap..."
                )
            }
        else:
            # Nothing recognised — show list clearly and ask again
            careers_list = "\n".join(f"  • {c}" for c in self.available_careers)
            return {
                "action": "ask_for_career",
                "skills": self.detected_skills,
                "message": (
                    f"I couldn't recognise '{user_input}' as a career.\n\n"
                    f"Please choose one from the list:\n{careers_list}"
                )
            }

    def _process_career_detected(self, user_input: str, skills: List[str]) -> Dict:
        if any(word in user_input for word in ['yes', 'yeah', 'yep', 'correct', 'right']):
            self.state = UserInteractionState.ASKING_SKILLS
            return {
                "action": "ask_for_skills",
                "career": self.detected_career,
                "message": self._generate_skills_prompt(self.detected_career)
            }
        elif any(word in user_input for word in ['no', 'nope', 'wrong', 'not']):
            self.state = UserInteractionState.INITIAL
            return {
                "action": "ask_for_career",
                "message": f"Sorry about that! What career are you interested in?\nAvailable: {', '.join(self.available_careers)}"
            }
        else:
            if skills:
                self.detected_skills = skills
                self.state = UserInteractionState.SKILLS_PROVIDED
                return {
                    "action": "confirm_skills",
                    "career": self.detected_career,
                    "skills": skills,
                    "message": f"For the {self.detected_career} path, I see you know: {', '.join(skills)}"
                }
            else:
                return self._process_initial_input(self._extract_career_from_text(user_input), [])

    def _get_skills_examples(self) -> str:
        """Return recognisable skill examples relevant to the detected career."""
        examples_map = {
            "frontend developer":  "HTML, CSS, JavaScript, React, TypeScript, Bootstrap",
            "backend developer":   "Python, Java, SQL, Node.js, Docker, Spring",
            "data scientist":      "Python, SQL, Statistics, Machine Learning, Pandas, NumPy",
            "ai engineer":         "Python, Machine Learning, Deep Learning, NLP, Computer Vision",
            "cybersecurity":       "Networking, Linux, Python, Ethical Hacking, Cryptography",
            "devops engineer":     "Docker, Linux, AWS, Git, Kubernetes",
            "mobile developer":    "Flutter, React Native, Swift, Kotlin, Java",
            "fullstack developer": "HTML, CSS, JavaScript, React, Node.js, SQL",
            "data engineer":       "Python, SQL, Spark, Hadoop, Airflow",
            "game developer":      "C++, Unity, Python, OpenGL",
        }
        career  = (self.detected_career or "").lower()
        examples = examples_map.get(career, "Python, SQL, JavaScript, HTML, CSS, React")
        return f"Examples for {self.detected_career or 'your career'}: {examples}"

    def _process_skills_response(self, user_input: str, extracted_skills: List[str]) -> Dict:
        none_indicators = [
            'none','no','nothing','n/a','na','nil','null','zero','0',
            'starting from scratch','beginner','no experience','no skills',
            "i don't know anything",'fresh','newbie','novice','no prior knowledge',
            'complete beginner','total beginner','just starting'
        ]

        # User explicitly says no skills
        if any(ind in user_input for ind in none_indicators):
            self._skills_retry = 0
            self.detected_skills = []
            self.experience_level = "beginner"
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_no_skills",
                "career": self.detected_career,
                "message": "No prior skills — that's totally fine! I'll create a complete beginner roadmap."
            }

        # Real skills recognised from the known skills list → accept them
        if extracted_skills:
            self._skills_retry = 0
            self.detected_skills = extracted_skills
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_skills",
                "career": self.detected_career,
                "skills": extracted_skills,
                "message": f"Got it! For {self.detected_career}, you know: {', '.join(extracted_skills)}"
            }

        # Nothing recognised — track retries to avoid infinite loop
        self._skills_retry = getattr(self, "_skills_retry", 0) + 1

        if self._skills_retry >= 2:
            # After 2 failed attempts treat as beginner and continue
            self._skills_retry = 0
            self.detected_skills = []
            self.experience_level = "beginner"
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_no_skills",
                "career": self.detected_career,
                "message": (
                    "I still couldn't recognise any skills — no worries! "
                    "I'll build a complete beginner roadmap for you."
                )
            }

        # First failed attempt → ask again with clear examples
        return {
            "action": "ask_skills_again",
            "career": self.detected_career,
            "message": (
                f"⚠️ I couldn't recognise \"{user_input}\" as a skill.\n\n"
                f"Please enter skills separated by commas.\n"
                f"{self._get_skills_examples()}\n\n"
                f"Or type 'none' if you're starting from scratch."
            )
        }

    def _process_experience_response(self, user_input: str) -> Dict:
        if any(w in user_input for w in ['beginner','new','starting','none','basic']):
            self.experience_level = "beginner"
        elif any(w in user_input for w in ['intermediate','some','moderate','familiar']):
            self.experience_level = "intermediate"
        elif any(w in user_input for w in ['advanced','expert','senior','experienced']):
            self.experience_level = "advanced"
        else:
            self.experience_level = "beginner"
        self.state = UserInteractionState.READY_TO_GENERATE
        return {"action": "generate_roadmap"}

    def _generate_skills_prompt(self, career: str) -> str:
        prompts = [
            f"Before I create your {career} roadmap, do you have any existing skills?\nFor example: Python, SQL, JavaScript.\nOr type 'none' if starting from scratch:",
            f"What skills do you already have that might help with {career}?\n(If you're a complete beginner, just say 'none'):",
            f"To personalize your {career} learning path, what skills do you currently know?\n(No skills yet? Just say 'none'):"
        ]
        return random.choice(prompts)

    def _parse_manual_skills(self, text: str) -> List[str]:
        text = re.sub(r'\b(i|we|have|know|can|do|some|basic|of|the|in|a|an)\b', '', text, flags=re.IGNORECASE)
        parts = re.split(r'[,;]|\band\b|\bwith\b|\balso\b', text)
        return [s.strip().lower() for s in parts if len(s.strip()) > 1 and s.strip().lower() not in ['none','no','n/a','nothing']]

    def _extract_career_from_text(self, text: str) -> Optional[str]:
        tl = text.lower()
        for career in self.available_careers:
            if career in tl:
                return career
        return None

    def reset(self):
        self.state = UserInteractionState.INITIAL
        self.detected_career = None
        self.detected_skills = []
        self.experience_level = "beginner"
        self.conversation_history = []

# =============================================================================
# ROADMAP GENERATOR  (identical logic, text_generator calls have static fallbacks)
# =============================================================================
class RoadmapGenerator:
    def __init__(self, model_manager: ModelManager, skills_df: pd.DataFrame):
        self.models       = model_manager
        self.skills_df    = skills_df
        self.career_database = self._build_career_database()

    def _build_career_database(self) -> Dict[str, CareerPath]:
        careers = {}
        for subject in self.skills_df['subject'].unique():
            subject_skills = self.skills_df[self.skills_df['subject'] == subject]
            skills = []
            for _, row in subject_skills.iterrows():
                prereqs = []
                if pd.notna(row.get('prerequisite')) and str(row['prerequisite']).lower() != 'none':
                    prereqs = [p.strip() for p in str(row['prerequisite']).split(',')]
                skill = Skill(
                    name=row['skill'],
                    level=row.get('level', 'intermediate'),
                    duration_weeks=int(row.get('duration_weeks', 2)),
                    prerequisites=prereqs,
                    category=self._categorize_skill(row['skill']),
                    importance=self._calculate_importance(row),
                    real_world_applications=self._get_applications(row['skill'])
                )
                skills.append(skill)
            careers[subject] = CareerPath(
                name=subject,
                description=self._generate_career_description(subject),
                required_skills=skills,
                market_demand=self._get_market_demand(subject),
                average_salary_range=self._get_salary_range(subject),
                industry_sectors=self._get_industry_sectors(subject)
            )
        return careers

    def generate_roadmap(self, subject: str, user_skills: List[str],
                         experience_level: str = "beginner") -> Dict:
        if subject not in self.career_database:
            return {"error": f"Career path '{subject}' not found"}

        career = self.career_database[subject]
        user_skills_set = set(user_skills)
        skills_to_learn, known_skills = [], []

        for skill in career.required_skills:
            if skill.name in user_skills_set or self._is_skill_known(skill.name, user_skills_set):
                known_skills.append(skill)
            else:
                skills_to_learn.append(skill)

        roadmap = self._build_progression(skills_to_learn, known_skills, experience_level)
        roadmap  = self._add_ai_explanations(roadmap, career)
        total_weeks = sum(s['duration_weeks'] for s in roadmap)

        return {
            "career": career.name,
            "description": career.description,
            "market_info": {
                "demand": career.market_demand,
                "salary": career.average_salary_range,
                "industries": career.industry_sectors
            },
            "known_skills": [
                {"name": s.name, "contribution": self._assess_skill_contribution(s.name, career.name)}
                for s in known_skills
            ],
            "roadmap": roadmap,
            "total_duration_weeks": total_weeks,
            "milestones": self._generate_milestones(roadmap),
            "job_readiness_score": self._calculate_readiness(len(known_skills), len(career.required_skills)),
            "is_beginner": len(user_skills) == 0
        }

    def _build_progression(self, skills_to_learn, known_skills, experience_level):
        roadmap = []
        completed = {s.name for s in known_skills}
        remaining = skills_to_learn.copy()
        iteration, max_iter = 0, len(remaining) * 2

        while remaining and iteration < max_iter:
            iteration += 1
            added = False
            for skill in remaining[:]:
                if all(p in completed for p in skill.prerequisites):
                    roadmap.append({
                        "skill": skill.name, "level": skill.level,
                        "duration_weeks": skill.duration_weeks,
                        "category": skill.category, "importance": skill.importance,
                        "prerequisites_met": skill.prerequisites,
                        "real_world_use": skill.real_world_applications,
                        "order": len(roadmap) + 1
                    })
                    completed.add(skill.name)
                    remaining.remove(skill)
                    added = True
            if not added and remaining:
                remaining.sort(key=lambda x: len([p for p in x.prerequisites if p not in completed]))
                ns = remaining.pop(0)
                roadmap.append({
                    "skill": ns.name, "level": ns.level,
                    "duration_weeks": ns.duration_weeks,
                    "category": ns.category, "importance": ns.importance,
                    "prerequisites_met": [p for p in ns.prerequisites if p in completed],
                    "missing_prerequisites": [p for p in ns.prerequisites if p not in completed],
                    "real_world_use": ns.real_world_applications,
                    "order": len(roadmap) + 1,
                    "warning": "Some prerequisites may not be fully met"
                })
                completed.add(ns.name)

        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        roadmap.sort(key=lambda x: (level_order.get(x["level"], 1), -x["importance"]))
        for i, step in enumerate(roadmap):
            step["order"] = i + 1
        return roadmap

    def _add_ai_explanations(self, roadmap, career):
        for step in roadmap:
            step["ai_explanation"]    = self._generate_skill_explanation(step["skill"], step["level"], career.name)
            step["learning_strategy"] = self._suggest_learning_strategy(step["skill"], step["level"])
            step["project_ideas"]     = self._generate_project_ideas(step["skill"], career.name)
        return roadmap

    def _generate_skill_explanation(self, skill: str, level: str, career: str) -> str:
        # text_generator stub returns None → use static template (same as original fallback)
        return f"{skill.title()} is essential for building expertise as a {career}"

    def _suggest_learning_strategy(self, skill: str, level: str) -> str:
        if level == "beginner":
            return "Start with interactive tutorials and hands-on exercises"
        elif level == "intermediate":
            return "Build small projects and participate in coding challenges"
        else:
            return "Contribute to open-source projects and tackle complex problems"

    def _generate_project_ideas(self, skill: str, career: str) -> List[str]:
        ideas_map = {
            "python":           ["Build a CLI automation tool", "Create a data analysis script"],
            "machine learning": ["Train a classification model on a public dataset", "Build a recommendation system"],
            "deep learning":    ["Image classifier with CNN", "Text sentiment analysis"],
            "sql":              ["Design a database schema for an e-commerce app", "Write complex JOIN queries"],
            "html":             ["Build a personal portfolio page", "Create a responsive landing page"],
            "css":              ["Style a blog layout", "Recreate a popular website UI"],
            "javascript":       ["Build a to-do list app", "Create a weather dashboard using an API"],
            "react":            ["Build a movie search app", "Create a dashboard with charts"],
            "docker":           ["Containerize a Python app", "Set up a multi-service docker-compose"],
            "networking":       ["Set up a home lab network", "Configure a firewall"],
            "linux":            ["Automate tasks with bash scripts", "Set up a web server"],
        }
        skill_lower = skill.lower()
        for key, ideas in ideas_map.items():
            if key in skill_lower:
                return ideas
        return [f"Build a portfolio project using {skill}", f"Solve real problems with {skill} in a {career} context"]

    def _categorize_skill(self, skill_name: str) -> str:
        categories = {
            "programming": ["python","java","javascript","typescript","c++","rust","go"],
            "data":        ["sql","pandas","numpy","excel","tableau","power bi"],
            "ml_ai":       ["machine learning","deep learning","nlp","computer vision"],
            "web":         ["html","css","react","angular","vue","node"],
            "devops":      ["docker","kubernetes","jenkins","git","aws","azure"],
            "mobile":      ["swift","kotlin","react native","flutter"],
            "security":    ["cybersecurity","penetration testing","encryption"]
        }
        for cat, skills in categories.items():
            if any(s in skill_name.lower() for s in skills):
                return cat
        return "general"

    def _calculate_importance(self, row: pd.Series) -> float:
        importance = 0.5
        if row.get('level') == 'beginner':
            importance += 0.2
        skill_name = row['skill']
        dependents = self.skills_df[self.skills_df['prerequisite'].str.contains(re.escape(skill_name), na=False, regex=True)]
        importance += min(len(dependents) * 0.1, 0.3)
        return min(importance, 1.0)

    def _get_applications(self, skill: str) -> List[str]:
        apps_map = {
            "python":           ["Data analysis","Web development","Automation","AI/ML models"],
            "machine learning": ["Predictive models","Recommendation systems","Fraud detection"],
            "deep learning":    ["Image recognition","NLP","Autonomous systems"],
            "sql":              ["Database management","Data querying","Business intelligence"],
        }
        for key, apps in apps_map.items():
            if key in skill.lower():
                return apps
        return ["Industry-specific applications","Personal projects","Professional development"]

    def _generate_career_description(self, subject: str) -> str:
        # static fallback (same as original's except block)
        return f"A {subject} designs and implements solutions using cutting-edge technologies"

    def _get_market_demand(self, subject: str) -> str:
        demand = {"ai engineer":"Very High","data scientist":"High","frontend developer":"High","cybersecurity":"Very High"}
        return demand.get(subject.lower(), "Moderate to High")

    def _get_salary_range(self, subject: str) -> str:
        salary = {
            "ai engineer":"$120,000 - $200,000","data scientist":"$100,000 - $180,000",
            "frontend developer":"$80,000 - $150,000","cybersecurity":"$90,000 - $170,000"
        }
        return salary.get(subject.lower(), "$70,000 - $150,000")

    def _get_industry_sectors(self, subject: str) -> List[str]:
        sectors = {
            "ai engineer":        ["Technology","Healthcare","Finance","Automotive"],
            "data scientist":     ["Technology","Finance","Healthcare","E-commerce"],
            "cybersecurity":      ["Government","Finance","Technology","Healthcare"]
        }
        return sectors.get(subject.lower(), ["Technology","Finance","Healthcare","Education"])

    def _is_skill_known(self, skill: str, user_skills: set) -> bool:
        if skill in user_skills:
            return True
        return any(rs in user_skills for rs in self._get_related_skills(skill))

    def _get_related_skills(self, skill: str) -> List[str]:
        relations = {
            "python":           ["programming","coding","scripting"],
            "javascript":       ["web development","frontend"],
            "machine learning": ["ai","artificial intelligence","deep learning"]
        }
        return relations.get(skill.lower(), [])

    def _assess_skill_contribution(self, skill: str, career: str) -> str:
        return f"Your {skill} knowledge provides a strong foundation for {career} concepts"

    def _generate_milestones(self, roadmap: List[Dict]) -> List[Dict]:
        milestones = []
        total = len(roadmap)
        if total > 0:
            milestones.append({"milestone":"Foundation Complete","after_step":min(3,total),"achievement":"You've built the core fundamentals!"})
        if total > 6:
            milestones.append({"milestone":"Intermediate Mastery","after_step":total//2,"achievement":"You can now build real-world projects!"})
        milestones.append({"milestone":"Career Ready","after_step":total,"achievement":"You're ready to apply for jobs!"})
        return milestones

    def _calculate_readiness(self, known: int, total: int) -> int:
        return 0 if total == 0 else min(100, int((known/total)*100))

    def check_skill_relevance(self, career: str, user_skills: List[str]) -> Dict:
        """
        Checks how many of the user's skills actually belong to the target career.
        Returns a dict with:
          - relevant_skills:   skills that match the career
          - irrelevant_skills: skills the user has but don't belong to this career
          - relevance_score:   0.0 – 1.0  (1.0 = all skills are relevant)
          - is_career_beginner: True if relevance_score < 0.3
          - message:           human-readable explanation
        """
        if not user_skills:
            return {
                "relevant_skills":   [],
                "irrelevant_skills": [],
                "relevance_score":   0.0,
                "is_career_beginner": True,
                "message": "No prior skills — building a complete beginner roadmap."
            }

        if career not in self.career_database:
            return {
                "relevant_skills":   [],
                "irrelevant_skills": user_skills,
                "relevance_score":   0.0,
                "is_career_beginner": True,
                "message": f"Career '{career}' not found."
            }

        # Collect all skill names for this career (exact + related)
        career_path   = self.career_database[career]
        career_skills = {s.name.lower() for s in career_path.required_skills}

        # Also pull in skills from OTHER careers to know what is "foreign"
        all_career_skills: Dict[str, set] = {}
        for c, cp in self.career_database.items():
            all_career_skills[c] = {s.name.lower() for s in cp.required_skills}

        relevant   = []
        irrelevant = []

        for skill in user_skills:
            skill_lower = skill.lower()
            # Direct match
            if skill_lower in career_skills:
                relevant.append(skill)
                continue
            # Related match — partial word overlap with a career skill
            if any(skill_lower in cs or cs in skill_lower for cs in career_skills):
                relevant.append(skill)
                continue
            irrelevant.append(skill)

        total          = len(user_skills)
        n_relevant     = len(relevant)
        relevance_score = n_relevant / total if total > 0 else 0.0

        # Classify: <30% relevant → treat as career beginner
        is_career_beginner = relevance_score < 0.3

        if is_career_beginner and irrelevant:
            other_careers = []
            for skill in irrelevant:
                for c, c_skills in all_career_skills.items():
                    if c != career and (skill.lower() in c_skills or
                       any(skill.lower() in cs or cs in skill.lower() for cs in c_skills)):
                        other_careers.append(c)
            other_careers = list(dict.fromkeys(other_careers))  # deduplicate

            if relevant:
                msg = (
                    f"You know {', '.join(relevant)} which will help with {career}. "
                    f"However, {', '.join(irrelevant)} "
                    f"{'belongs' if len(irrelevant)==1 else 'belong'} to a different field"
                    f"{' (' + ', '.join(other_careers[:2]) + ')' if other_careers else ''}. "
                    f"I'll build your roadmap from scratch for {career}, "
                    f"but your existing skills will still give you a head start!"
                )
            else:
                msg = (
                    f"Your skills ({', '.join(irrelevant)}) "
                    f"{'belongs' if len(irrelevant)==1 else 'belong'} to a different field"
                    f"{' (' + ', '.join(other_careers[:2]) + ')' if other_careers else ''}, "
                    f"not to {career}. "
                    f"No worries — I'll create a complete beginner roadmap for {career} for you!"
                )
        elif not is_career_beginner and irrelevant:
            msg = (
                f"Most of your skills ({', '.join(relevant)}) are relevant to {career}. "
                f"{', '.join(irrelevant)} "
                f"{'is' if len(irrelevant)==1 else 'are'} from a different area "
                f"but won't hold you back!"
            )
        else:
            msg = (
                f"Your skills ({', '.join(relevant)}) are a great match for {career}. "
                f"I'll build on what you already know!"
            )

        return {
            "relevant_skills":    relevant,
            "irrelevant_skills":  irrelevant,
            "relevance_score":    round(relevance_score, 2),
            "is_career_beginner": is_career_beginner,
            "message":            msg
        }

# =============================================================================
# CAREER EXTRACTOR  (pure patterns — replaces bart-large-mnli)
# =============================================================================
CAREER_PATTERNS = {
    "ai engineer":          ["ai engineer","ai engineering","ml engineer","machine learning engineer","artificial intelligence engineer"],
    "data scientist":       ["data scientist","data science"],
    "frontend developer":   ["frontend","front end","frontend developer","front-end developer","ui developer"],
    "cybersecurity":        ["cybersecurity","cyber security","security engineer","ethical hacker","penetration tester"],
    "data engineer":        ["data engineer","data engineering"],
    "devops engineer":      ["devops","devops engineer","site reliability","sre"],
    "mobile developer":     ["mobile developer","mobile dev","ios developer","android developer"],
    "game developer":       ["game developer","game dev","game programming","game designer"],
    "backend developer":    ["backend","back end","backend developer","back-end developer","server side"],
    "fullstack developer":  ["fullstack","full stack","full-stack developer"],
}

def extract_career(text: str) -> Optional[str]:
    tl = text.lower()
    for career, patterns in CAREER_PATTERNS.items():
        if any(p in tl for p in patterns):
            return career
    return None

# =============================================================================
# STARTUP — load data and initialise components
# =============================================================================
print("📦 Loading data and initialising components...")

def load_skills_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"⚠️  {path} not found — using demo data")
        df = pd.DataFrame({
            'subject': [
                'ai engineer','ai engineer','ai engineer','ai engineer',
                'ai engineer','ai engineer','ai engineer','ai engineer',
                'data scientist','data scientist','data scientist','data scientist','data scientist','data scientist',
                'frontend developer','frontend developer','frontend developer','frontend developer','frontend developer',
                'cybersecurity','cybersecurity','cybersecurity','cybersecurity','cybersecurity',
            ],
            'skill': [
                'python','numpy','pandas','machine learning',
                'deep learning','natural language processing','computer vision','reinforcement learning',
                'python','sql','statistics','machine learning','data visualization','big data',
                'html','css','javascript','react','typescript',
                'networking','linux','python','ethical hacking','cryptography',
            ],
            'level': [
                'beginner','beginner','beginner','intermediate',
                'advanced','advanced','advanced','expert',
                'beginner','beginner','intermediate','intermediate','intermediate','advanced',
                'beginner','beginner','intermediate','intermediate','advanced',
                'beginner','beginner','intermediate','advanced','advanced',
            ],
            'duration_weeks': [
                4,2,2,6,8,6,6,8,
                4,2,3,6,3,6,
                2,2,4,6,4,
                3,3,4,8,6,
            ],
            'prerequisite': [
                'none','python','python','python,numpy',
                'machine learning','python,deep learning','python,deep learning','deep learning,machine learning',
                'none','none','python','python,sql,statistics','python,pandas','python,sql',
                'none','none','html,css','javascript','javascript,react',
                'none','networking','python,linux','networking,linux','networking,python',
            ]
        })

    required_cols = ['subject','skill','level','duration_weeks']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 2 if col == 'duration_weeks' else 'intermediate'

    df = df.fillna({'prerequisite':'none','duration_weeks':2,'level':'intermediate'})
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df

skills_df         = load_skills_data(SKILLS_PATH)
available_careers = skills_df['subject'].unique().tolist()
model_manager     = ModelManager()
skill_extractor   = SkillExtractor(model_manager, skills_df['skill'].unique().tolist())
roadmap_generator = RoadmapGenerator(model_manager, skills_df)

# In-memory session store
sessions: Dict = {}

def get_session(sid: str) -> Dict:
    if sid not in sessions:
        sessions[sid] = {"dialogue": DialogueManager(available_careers)}
    return sessions[sid]

print(f"Ready! {len(skills_df)} skills across {len(available_careers)} career paths")
print(f"Careers: {', '.join(available_careers)}\n")

# =============================================================================
# SCHEMAS
# =============================================================================
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class RoadmapRequest(BaseModel):
    career:           str
    known_skills:     List[str] = []
    experience_level: str       = "beginner"

class ResetRequest(BaseModel):
    session_id: str = "default"

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["General"])
def root():
    return {
        "status": "running 🚀",
        "docs": "/docs",
        "available_careers": available_careers,
        "endpoints": ["/welcome","/careers","/chat","/roadmap","/reset"]
    }

@app.get("/welcome", tags=["General"])
def welcome():
    return {
        "message": "🚀 AI-Powered Career Roadmap Generator",
        "instructions": "Tell me your career goal and skills. I'll build a personalised learning path.",
        "examples": [
            "I want to become an AI Engineer",
            "Data Scientist with Python and SQL skills",
            "Frontend Developer starting from scratch"
        ]
    }

@app.get("/careers", tags=["General"])
def get_careers():
    return {"careers": available_careers, "total": len(available_careers)}

@app.post("/roadmap", tags=["Roadmap"])
def generate_roadmap(req: RoadmapRequest):
    """
    Stateless one-shot roadmap generation.
    Pass career name, your known skills, and experience level.
    """
    result = roadmap_generator.generate_roadmap(
        req.career.lower(), req.known_skills, req.experience_level
    )
    return result

@app.post("/chat", tags=["Chat"])
def chat(req: ChatRequest):
    """
    Stateful conversational endpoint.
    Send messages naturally; the bot guides you through career → skills → roadmap.
    When step == 'roadmap_ready', the 'roadmap' key contains your full learning path.
    """
    sid     = req.session_id
    message = req.message.strip()

    if not message:
        return {"reply": "Please enter a message.", "step": "start", "session_id": sid}

    session  = get_session(sid)
    dialogue = session["dialogue"]

    detected_career = extract_career(message)
    detected_skills = skill_extractor.extract(message)

    result = dialogue.process_input(message, detected_career, detected_skills)
    action = result.get("action")

    # ── Exit ──
    if action == "exit":
        sessions.pop(sid, None)
        return {"reply": "👋 Good luck on your learning journey! Come back anytime.", "step": "exit", "session_id": sid}

    # ── Restart ──
    if action == "restart":
        return {"reply": "🔄 Starting fresh! Tell me about your career goals.", "step": "start", "session_id": sid}

    # ── Need more info ──
    if action in ("ask_for_skills", "ask_for_career", "ask_everything", "ask_skills_again"):
        return {"reply": result.get("message",""), "step": action, "session_id": sid,
                "career": result.get("career"), "detected_skills": result.get("skills",[])}

    # ── Confirmations + Generate (shared logic) ──────────────────────────────
    if action in ("confirm_career_and_skills", "confirm_skills",
                  "confirm_no_skills", "generate_roadmap"):

        career_name = dialogue.detected_career
        raw_skills  = dialogue.detected_skills
        exp_level   = dialogue.experience_level

        # ── Skill relevance check ─────────────────────────────────────────────
        relevance = roadmap_generator.check_skill_relevance(career_name, raw_skills)

        if relevance["is_career_beginner"]:
            # Skills don't belong to this career → build full beginner roadmap
            # Pass empty skills so ALL career skills appear in the roadmap
            skills_for_roadmap = []
            exp_level          = "beginner"
        else:
            # Skills are relevant → credit them, skip those steps in roadmap
            skills_for_roadmap = relevance["relevant_skills"]

        roadmap = roadmap_generator.generate_roadmap(
            career_name, skills_for_roadmap, exp_level
        )
        dialogue.reset()

        # Build the reply message
        if not raw_skills:
            reply = (
                f"Here is your complete beginner roadmap for {career_name}!"
            )
        elif relevance["is_career_beginner"]:
            reply = (
                f"{relevance['message']}\n\n"
                f"Here is your full roadmap for {career_name} from the beginning:"
            )
        else:
            reply = (
                f"{relevance['message']}\n\n"
                f"Here is your personalised roadmap for {career_name}:"
            )

        return {
            "reply":             reply,
            "step":              "roadmap_ready",
            "session_id":        sid,
            "career":            career_name,
            "known_skills":      raw_skills,
            "relevant_skills":   relevance["relevant_skills"],
            "irrelevant_skills": relevance["irrelevant_skills"],
            "relevance_score":   relevance["relevance_score"],
            "is_beginner_path":  relevance["is_career_beginner"],
            "roadmap":           roadmap
        }

    return {"reply": result.get("message", "How can I help?"), "step": action, "session_id": sid}

@app.post("/reset", tags=["General"])
def reset(req: ResetRequest):
    sessions.pop(req.session_id, None)
    return {"message": f"Session '{req.session_id}' reset.", "status": "ok"}
