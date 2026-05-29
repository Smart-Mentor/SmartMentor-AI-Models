import pandas as pd
import re
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Modern local AI - no external APIs needed
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    AutoModelForSequenceClassification
)
import torch
from sentence_transformers import SentenceTransformer

# =========================
# ENHANCED DATA CLASSES
# =========================
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
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class UserInteractionState(Enum):
    INITIAL = "initial"
    CAREER_DETECTED = "career_detected"
    ASKING_SKILLS = "asking_skills"
    SKILLS_PROVIDED = "skills_provided"
    READY_TO_GENERATE = "ready_to_generate"

# =========================
# DIALOGUE MANAGER - NEW!
# =========================
class DialogueManager:
    """Manages conversation flow and ensures all necessary information is gathered"""
    
    def __init__(self, available_careers: List[str]):
        self.state = UserInteractionState.INITIAL
        self.available_careers = available_careers
        self.detected_career = None
        self.detected_skills = []
        self.experience_level = "beginner"
        self.conversation_history = []
    
    def process_input(self, user_input: str, extracted_career: Optional[str], 
                     extracted_skills: List[str]) -> Dict:
        """
        Process user input based on current conversation state.
        Returns dict with action to take next.
        """
        user_input_lower = user_input.lower().strip()
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Check for exit commands
        if user_input_lower in ['quit', 'exit', 'q', 'stop']:
            return {"action": "exit"}
        
        # Check for restart command
        if user_input_lower in ['restart', 'reset', 'start over']:
            self.reset()
            return {"action": "restart"}
        
        # Process based on current state
        if self.state == UserInteractionState.INITIAL:
            return self._process_initial_input(extracted_career, extracted_skills)
        
        elif self.state == UserInteractionState.CAREER_DETECTED:
            return self._process_career_detected(user_input_lower, extracted_skills)
        
        elif self.state == UserInteractionState.ASKING_SKILLS:
            return self._process_skills_response(user_input_lower, extracted_skills)
        
        elif self.state == UserInteractionState.SKILLS_PROVIDED:
            return self._process_experience_response(user_input_lower)
        
        else:
            return {"action": "generate_roadmap"}
    
    def _process_initial_input(self, career: Optional[str], skills: List[str]) -> Dict:
        """Process initial user input"""
        self.detected_career = career
        self.detected_skills = skills
        
        # Case 1: Both career and skills detected
        if career and skills:
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_career_and_skills",
                "career": career,
                "skills": skills,
                "message": f"I detected you want to become a {career} and know {', '.join(skills)}"
            }
        
        # Case 2: Only career detected - NEED TO ASK FOR SKILLS
        elif career and not skills:
            self.state = UserInteractionState.ASKING_SKILLS
            return {
                "action": "ask_for_skills",
                "career": career,
                "message": self._generate_skills_prompt(career)
            }
        
        # Case 3: Only skills detected - need to ask for career
        elif not career and skills:
            return {
                "action": "ask_for_career",
                "skills": skills,
                "message": f"Sorry about that! What career are you interested in?"
            }
        
        # Case 4: Nothing detected
        else:
            return {
                "action": "ask_everything",
                "message": "I couldn't detect your career goal. Could you tell me:\n1. What career are you aiming for?\n2. Any skills you already have?"
            }
    
    def _process_career_detected(self, user_input: str, skills: List[str]) -> Dict:
        """Process input when career has been detected but skills unknown"""
        # Check if user is providing skills or answering a different question
        if any(word in user_input for word in ['yes', 'yeah', 'yep', 'correct', 'right']):
            # User confirmed career, now ask about skills
            self.state = UserInteractionState.ASKING_SKILLS
            return {
                "action": "ask_for_skills",
                "career": self.detected_career,
                "message": self._generate_skills_prompt(self.detected_career)
            }
        
        elif any(word in user_input for word in ['no', 'nope', 'wrong', 'not']):
            # User says career is wrong, ask them to specify
            self.state = UserInteractionState.INITIAL
            return {
                "action": "ask_for_career",
                "message": f"Sorry about that! What career are you interested in?\nAvailable options: {', '.join(self.available_careers)}"
            }
        
        else:
            # User might be providing skills or changing career
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
                # Check if they're providing a new career
                return self._process_initial_input(
                    self._extract_career_from_text(user_input),
                    []
                )
    
    def _process_skills_response(self, user_input: str, extracted_skills: List[str]) -> Dict:
        """Process user's response about skills"""
        # Check for variations of "none" or "no skills"
        none_indicators = [
            'none', 'no', 'nothing', 'n/a', 'na', 'nil', 'null',
            'zero', '0', 'starting from scratch', 'beginner',
            'no experience', 'no skills', 'i don\'t know anything',
            'fresh', 'newbie', 'novice', 'no prior knowledge',
            'complete beginner', 'total beginner', 'just starting'
        ]
        
        user_lower = user_input.lower()
        
        # Check if user has no skills
        if any(indicator in user_lower for indicator in none_indicators) or \
           (len(user_input.strip()) < 10 and not extracted_skills):
            self.detected_skills = []
            self.experience_level = "beginner"
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_no_skills",
                "career": self.detected_career,
                "message": "No prior skills - that's totally fine! I'll create a complete beginner roadmap for you."
            }
        
        # User provided some skills
        if extracted_skills:
            self.detected_skills = extracted_skills
            self.state = UserInteractionState.SKILLS_PROVIDED
            return {
                "action": "confirm_skills",
                "career": self.detected_career,
                "skills": extracted_skills,
                "message": f"Got it! For {self.detected_career}, you know: {', '.join(extracted_skills)}"
            }
        else:
            # Try to parse skills from text manually
            potential_skills = self._parse_manual_skills(user_input)
            if potential_skills:
                self.detected_skills = potential_skills
                self.state = UserInteractionState.SKILLS_PROVIDED
                return {
                    "action": "confirm_skills",
                    "career": self.detected_career,
                    "skills": potential_skills,
                    "message": f"I understood these skills: {', '.join(potential_skills)}"
                }
            else:
                # Still unclear - ask more specifically
                return {
                    "action": "ask_skills_again",
                    "career": self.detected_career,
                    "message": "I want to make sure I understand. Could you list the skills you know?\nFor example: 'Python, SQL, basic statistics'\nOr type 'none' if you're starting from scratch."
                }
    
    def _process_experience_response(self, user_input: str) -> Dict:
        """Process response about experience level"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['beginner', 'new', 'starting', 'none', 'basic']):
            self.experience_level = "beginner"
        elif any(word in user_lower for word in ['intermediate', 'some', 'moderate', 'familiar']):
            self.experience_level = "intermediate"
        elif any(word in user_lower for word in ['advanced', 'expert', 'senior', 'experienced']):
            self.experience_level = "advanced"
        else:
            self.experience_level = "beginner"
        
        self.state = UserInteractionState.READY_TO_GENERATE
        return {"action": "generate_roadmap"}
    
    def _generate_skills_prompt(self, career: str) -> str:
        """Generate contextual prompt asking for skills"""
        prompts = [
            f"Before I create your {career} roadmap, do you have any existing skills?\nFor example: Python, SQL, JavaScript, etc.\nOr type 'none' if you're starting from scratch:",
            f"What skills do you already have that might help with {career}?\n(If you're a complete beginner, just say 'none'):",
            f"To personalize your {career} learning path, what skills do you currently know?\n(If you don't know any yet, that's perfectly fine - just say 'none'):"
        ]
        import random
        return random.choice(prompts)
    
    def _parse_manual_skills(self, text: str) -> List[str]:
        """Manual skill parsing from text"""
        # Remove common filler words
        text = re.sub(r'\b(i|we|have|know|can|do|some|basic|of|the|in|a|an)\b', '', text, flags=re.IGNORECASE)
        
        # Split by common separators
        skills = re.split(r'[,;]|\band\b|\bwith\b|\balso\b', text)
        
        # Clean and filter
        cleaned_skills = []
        for skill in skills:
            skill = skill.strip().lower()
            if len(skill) > 1 and skill not in ['none', 'no', 'n/a', 'nothing']:
                cleaned_skills.append(skill)
        
        return cleaned_skills
    
    def _extract_career_from_text(self, text: str) -> Optional[str]:
        """Try to extract career from text"""
        text_lower = text.lower()
        for career in self.available_careers:
            if career in text_lower:
                return career
        return None
    
    def reset(self):
        """Reset the dialogue state"""
        self.state = UserInteractionState.INITIAL
        self.detected_career = None
        self.detected_skills = []
        self.experience_level = "beginner"
        self.conversation_history = []

# =========================
# MODERN NLP MODEL MANAGER
# =========================
class ModelManager:
    """Manages multiple AI models for different tasks"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        # Load models lazily
        self._sentence_model = None
        self._intent_classifier = None
        self._text_generator = None
        self._ner_model = None
        self._skill_extractor = None
        
    @property
    def sentence_model(self):
        if self._sentence_model is None:
            print("📥 Loading semantic model...")
            self._sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
        return self._sentence_model
    
    @property
    def intent_classifier(self):
        if self._intent_classifier is None:
            print("📥 Loading intent classifier...")
            self._intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
        return self._intent_classifier
    
    @property
    def text_generator(self):
        if self._text_generator is None:
            print("📥 Loading text generator...")
            model_name = "google/flan-t5-base"
            
            self._text_generator = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
            )
        return self._text_generator
    
    @property
    def ner_model(self):
        if self._ner_model is None:
            print("📥 Loading NER model...")
            self._ner_model = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
        return self._ner_model
    
    @property
    def skill_extractor(self):
        if self._skill_extractor is None:
            print("📥 Loading skill extraction model...")
            model_name = "jjzha/jobbert-knowledge-extraction"
            try:
                self._skill_extractor = pipeline(
                    "token-classification",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
            except:
                self._skill_extractor = self.ner_model
        return self._skill_extractor

# =========================
# INTELLIGENT SKILL EXTRACTOR
# =========================
class SkillExtractor:
    """Advanced skill extraction using multiple AI models"""
    
    def __init__(self, model_manager: ModelManager, known_skills: List[str]):
        self.models = model_manager
        self.known_skills = set(known_skills)
        self.skill_embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Pre-compute embeddings for known skills"""
        if self.known_skills:
            self.skill_embeddings = self.models.sentence_model.encode(
                list(self.known_skills),
                convert_to_tensor=True
            )
    
    def extract(self, text: str) -> List[str]:
        """
        Multi-strategy skill extraction
        """
        if not text or len(text.strip()) < 3:
            return []
        
        text = text.lower().strip()
        extracted_skills = set()
        
        # Strategy 1: Direct pattern matching
        extracted_skills.update(self._pattern_match(text))
        
        # Strategy 2: Semantic similarity
        if len(text.split()) >= 2:
            extracted_skills.update(self._semantic_match(text))
        
        # Strategy 3: NER-based extraction
        if len(text.split()) > 3:
            extracted_skills.update(self._ner_extract(text))
        
        # Strategy 4: Context-aware extraction
        if len(text.split()) > 5:
            extracted_skills.update(self._contextual_extract(text))
        
        # Filter out generic words
        extracted_skills = self._filter_generic_terms(extracted_skills)
        
        return list(extracted_skills)
    
    def _pattern_match(self, text: str) -> List[str]:
        """Direct pattern matching"""
        matches = set()
        for skill in self.known_skills:
            if skill in text or re.search(r'\b' + re.escape(skill) + r'\b', text):
                matches.add(skill)
        return list(matches)
    
    def _semantic_match(self, text: str) -> List[str]:
        """Use embeddings for semantic similarity"""
        if self.skill_embeddings is None or len(self.known_skills) == 0:
            return []
        
        text_embedding = self.models.sentence_model.encode(text, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding.unsqueeze(0),
            self.skill_embeddings
        )
        
        threshold = 0.65
        matched_indices = torch.where(similarities > threshold)[0]
        return [list(self.known_skills)[i] for i in matched_indices]
    
    def _ner_extract(self, text: str) -> List[str]:
        """Extract skills using NER"""
        try:
            entities = self.models.ner_model(text)
            skills = []
            for entity in entities:
                if entity['entity_group'] in ['MISC', 'ORG']:
                    entity_text = entity['word'].lower()
                    for skill in self.known_skills:
                        if skill in entity_text or entity_text in skill:
                            skills.append(skill)
                            break
            return skills
        except:
            return []
    
    def _contextual_extract(self, text: str) -> List[str]:
        """Use zero-shot classification"""
        try:
            candidate_labels = [
                "programming languages",
                "frameworks", 
                "tools",
                "concepts",
                "technologies"
            ]
            
            result = self.models.intent_classifier(
                text,
                candidate_labels,
                multi_label=True
            )
            
            if result['scores'][0] > 0.5:
                words = text.split()
                potential_skills = []
                for i in range(len(words)):
                    for j in range(i+1, min(i+4, len(words)+1)):
                        phrase = ' '.join(words[i:j])
                        if len(phrase) > 2:
                            potential_skills.append(phrase)
                
                return [p for p in potential_skills if p in self.known_skills]
        except:
            pass
        
        return []
    
    def _filter_generic_terms(self, skills: set) -> set:
        """Remove overly generic terms"""
        generic_terms = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
            'should', 'now', 'also', 'been', 'being', 'have', 'has', 'had',
            'does', 'did', 'doing', 'would', 'could', 'this', 'that', 'these',
            'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she',
            'it', 'they', 'them', 'know', 'have', 'has', 'had', 'can', 'do',
            'want', 'become'
        }
        return {s for s in skills if s not in generic_terms and len(s) > 1}

# =========================
# ROADMAP GENERATOR
# =========================
class RoadmapGenerator:
    """AI-powered roadmap generation with realistic progression"""
    
    def __init__(self, model_manager: ModelManager, skills_df: pd.DataFrame):
        self.models = model_manager
        self.skills_df = skills_df
        self.career_database = self._build_career_database()
    
    def _build_career_database(self) -> Dict[str, CareerPath]:
        """Build structured career database from CSV"""
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
        """
        Generate intelligent roadmap
        """
        if subject not in self.career_database:
            return {"error": f"Career path '{subject}' not found"}
        
        career = self.career_database[subject]
        user_skills_set = set(user_skills)
        
        # Separate known and unknown skills
        skills_to_learn = []
        known_skills = []
        
        for skill in career.required_skills:
            if skill.name in user_skills_set or self._is_skill_known(skill.name, user_skills_set):
                known_skills.append(skill)
            else:
                skills_to_learn.append(skill)
        
        # Build learning progression
        roadmap = self._build_progression(skills_to_learn, known_skills, experience_level)
        
        # Add AI explanations
        roadmap_with_explanations = self._add_ai_explanations(roadmap, career)
        
        # Calculate timeline
        total_weeks = sum(step['duration_weeks'] for step in roadmap_with_explanations)
        
        return {
            "career": career.name,
            "description": career.description,
            "market_info": {
                "demand": career.market_demand,
                "salary": career.average_salary_range,
                "industries": career.industry_sectors
            },
            "known_skills": [
                {
                    "name": s.name,
                    "contribution": self._assess_skill_contribution(s.name, career.name)
                }
                for s in known_skills
            ],
            "roadmap": roadmap_with_explanations,
            "total_duration_weeks": total_weeks,
            "milestones": self._generate_milestones(roadmap_with_explanations),
            "job_readiness_score": self._calculate_readiness(len(known_skills), len(career.required_skills)),
            "is_beginner": len(user_skills) == 0
        }
    
    def _build_progression(self, skills_to_learn: List[Skill], 
                          known_skills: List[Skill],
                          experience_level: str) -> List[Dict]:
        """Build logical skill progression"""
        roadmap = []
        completed_skills = {s.name for s in known_skills}
        remaining_skills = skills_to_learn.copy()
        
        iteration = 0
        max_iterations = len(remaining_skills) * 2
        
        while remaining_skills and iteration < max_iterations:
            iteration += 1
            added_this_round = False
            
            for skill in remaining_skills[:]:
                if all(prereq in completed_skills for prereq in skill.prerequisites):
                    roadmap.append({
                        "skill": skill.name,
                        "level": skill.level,
                        "duration_weeks": skill.duration_weeks,
                        "category": skill.category,
                        "importance": skill.importance,
                        "prerequisites_met": skill.prerequisites,
                        "real_world_use": skill.real_world_applications,
                        "order": len(roadmap) + 1
                    })
                    completed_skills.add(skill.name)
                    remaining_skills.remove(skill)
                    added_this_round = True
            
            if not added_this_round and remaining_skills:
                remaining_skills.sort(key=lambda x: len([p for p in x.prerequisites if p not in completed_skills]))
                next_skill = remaining_skills.pop(0)
                roadmap.append({
                    "skill": next_skill.name,
                    "level": next_skill.level,
                    "duration_weeks": next_skill.duration_weeks,
                    "category": next_skill.category,
                    "importance": next_skill.importance,
                    "prerequisites_met": [p for p in next_skill.prerequisites if p in completed_skills],
                    "missing_prerequisites": [p for p in next_skill.prerequisites if p not in completed_skills],
                    "real_world_use": next_skill.real_world_applications,
                    "order": len(roadmap) + 1,
                    "warning": "Some prerequisites may not be fully met"
                })
                completed_skills.add(next_skill.name)
        
        # Sort by level and importance
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        roadmap.sort(key=lambda x: (level_order.get(x["level"], 1), -x["importance"]))
        
        for i, step in enumerate(roadmap):
            step["order"] = i + 1
        
        return roadmap
    
    def _add_ai_explanations(self, roadmap: List[Dict], career: CareerPath) -> List[Dict]:
        """Add AI-generated explanations"""
        enhanced_roadmap = []
        
        for step in roadmap:
            explanation = self._generate_skill_explanation(
                step["skill"], 
                step["level"],
                career.name
            )
            
            step["ai_explanation"] = explanation
            step["learning_strategy"] = self._suggest_learning_strategy(step["skill"], step["level"])
            step["project_ideas"] = self._generate_project_ideas(step["skill"], career.name)
            
            enhanced_roadmap.append(step)
        
        return enhanced_roadmap
    
    def _generate_skill_explanation(self, skill: str, level: str, career: str) -> str:
        """Use AI to generate personalized explanation"""
        prompt = f"Explain why {skill} is important for a {career} at {level} level. Keep it concise."
        
        try:
            response = self.models.text_generator(
                prompt,
                max_length=100,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']
            return response
        except:
            return f"{skill} is essential for building expertise as a {career}"
    
    def _suggest_learning_strategy(self, skill: str, level: str) -> str:
        """Suggest personalized learning strategy"""
        if level == "beginner":
            return "Start with interactive tutorials and hands-on exercises"
        elif level == "intermediate":
            return "Build small projects and participate in coding challenges"
        else:
            return "Contribute to open-source projects and tackle complex problems"
    
    def _generate_project_ideas(self, skill: str, career: str) -> List[str]:
        """Generate relevant project ideas"""
        prompt = f"Suggest 2 project ideas for {skill} as a {career}"
        
        try:
            response = self.models.text_generator(
                prompt,
                max_length=100,
                do_sample=True,
                temperature=0.8
            )[0]['generated_text']
            ideas = [idea.strip() for idea in response.split('.') if idea.strip()]
            return ideas[:2] if ideas else ["Build a small project using this skill"]
        except:
            return ["Create a portfolio project showcasing this skill"]
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize skills into domains"""
        categories = {
            "programming": ["python", "java", "javascript", "typescript", "c++", "rust", "go"],
            "data": ["sql", "pandas", "numpy", "excel", "tableau", "power bi"],
            "ml_ai": ["machine learning", "deep learning", "nlp", "computer vision"],
            "web": ["html", "css", "react", "angular", "vue", "node"],
            "devops": ["docker", "kubernetes", "jenkins", "git", "aws", "azure"],
            "mobile": ["swift", "kotlin", "react native", "flutter"],
            "security": ["cybersecurity", "penetration testing", "encryption"]
        }
        
        for category, skills in categories.items():
            if any(s in skill_name.lower() for s in skills):
                return category
        return "general"
    
    def _calculate_importance(self, row: pd.Series) -> float:
        """Calculate skill importance"""
        importance = 0.5
        if row.get('level') == 'beginner':
            importance += 0.2
        
        skill_name = row['skill']
        dependents = self.skills_df[
            self.skills_df['prerequisite'].str.contains(skill_name, na=False)
        ]
        importance += min(len(dependents) * 0.1, 0.3)
        
        return min(importance, 1.0)
    
    def _get_applications(self, skill: str) -> List[str]:
        """Get real-world applications"""
        applications_map = {
            "python": ["Data analysis", "Web development", "Automation scripts", "AI/ML models"],
            "machine learning": ["Predictive models", "Recommendation systems", "Fraud detection"],
            "deep learning": ["Image recognition", "Natural language processing", "Autonomous systems"],
            "sql": ["Database management", "Data querying", "Business intelligence"],
        }
        skill_lower = skill.lower()
        for key, apps in applications_map.items():
            if key in skill_lower:
                return apps
        return ["Industry-specific applications", "Personal projects", "Professional development"]
    
    def _generate_career_description(self, subject: str) -> str:
        """Generate career description"""
        try:
            prompt = f"Write a description of what a {subject} does"
            response = self.models.text_generator(prompt, max_length=100)[0]['generated_text']
            return response
        except:
            return f"A {subject} designs and implements solutions using cutting-edge technologies"
    
    def _get_market_demand(self, subject: str) -> str:
        """Get market demand"""
        demand_data = {
            "ai engineer": "Very High",
            "data scientist": "High",
            "frontend developer": "High",
            "cybersecurity": "Very High"
        }
        return demand_data.get(subject.lower(), "Moderate to High")
    
    def _get_salary_range(self, subject: str) -> str:
        """Get salary range"""
        salary_data = {
            "ai engineer": "$120,000 - $200,000",
            "data scientist": "$100,000 - $180,000",
            "frontend developer": "$80,000 - $150,000",
            "cybersecurity": "$90,000 - $170,000"
        }
        return salary_data.get(subject.lower(), "$70,000 - $150,000")
    
    def _get_industry_sectors(self, subject: str) -> List[str]:
        """Get industry sectors"""
        sectors = {
            "ai engineer": ["Technology", "Healthcare", "Finance", "Automotive"],
            "data scientist": ["Technology", "Finance", "Healthcare", "E-commerce"],
            "cybersecurity": ["Government", "Finance", "Technology", "Healthcare"]
        }
        return sectors.get(subject.lower(), ["Technology", "Finance", "Healthcare", "Education"])
    
    def _is_skill_known(self, skill: str, user_skills: set) -> bool:
        """Check if a skill is effectively known"""
        if skill in user_skills:
            return True
        related_skills = self._get_related_skills(skill)
        return any(rs in user_skills for rs in related_skills)
    
    def _get_related_skills(self, skill: str) -> List[str]:
        """Get related skills"""
        relations = {
            "python": ["programming", "coding", "scripting"],
            "javascript": ["web development", "frontend"],
            "machine learning": ["ai", "artificial intelligence", "deep learning"]
        }
        return relations.get(skill.lower(), [])
    
    def _assess_skill_contribution(self, skill: str, career: str) -> str:
        """Assess skill contribution"""
        return f"Your {skill} knowledge provides a strong foundation for {career} concepts"
    
    def _generate_milestones(self, roadmap: List[Dict]) -> List[Dict]:
        """Generate milestones"""
        milestones = []
        total_steps = len(roadmap)
        
        if total_steps > 0:
            milestones.append({
                "milestone": "Foundation Complete",
                "after_step": min(3, total_steps),
                "achievement": "You've built the core fundamentals!"
            })
        
        if total_steps > 6:
            milestones.append({
                "milestone": "Intermediate Mastery",
                "after_step": total_steps // 2,
                "achievement": "You can now build real-world projects!"
            })
        
        milestones.append({
            "milestone": "Career Ready",
            "after_step": total_steps,
            "achievement": "You're ready to apply for jobs!"
        })
        
        return milestones
    
    def _calculate_readiness(self, known_count: int, total_count: int) -> int:
        """Calculate readiness score"""
        if total_count == 0:
            return 0
        return min(100, int((known_count / total_count) * 100))

# =========================
# MODERN UI WITH DIALOGUE SUPPORT
# =========================
class ModernUI:
    """Enhanced user interface with dialogue support"""
    
    @staticmethod
    def display_welcome():
        """Display welcome screen"""
        print("\n" + "="*70)
        print("🚀  AI-POWERED CAREER ROADMAP GENERATOR  🚀")
        print("="*70)
        print("\n💡  Powered by modern local AI models - no internet required!")
        print("📚  Describe your career goal and I'll ask about your skills\n")
    
    @staticmethod
    def display_roadmap(roadmap_data: Dict):
        """Display roadmap with modern formatting"""
        if "error" in roadmap_data:
            print(f"\n❌ {roadmap_data['error']}")
            return
        
        print("\n" + "="*70)
        print(f"🎯  CAREER PATH: {roadmap_data['career'].upper()}")
        print("="*70)
        
        # Market info
        print(f"\n📊  MARKET OVERVIEW:")
        print(f"   • Demand: {roadmap_data['market_info']['demand']}")
        print(f"   • Salary Range: {roadmap_data['market_info']['salary']}")
        print(f"   • Industries: {', '.join(roadmap_data['market_info']['industries'])}")
        
        # Known skills
        if roadmap_data.get('known_skills'):
            print(f"\n✅  YOUR EXISTING SKILLS ({len(roadmap_data['known_skills'])}):")
            for skill in roadmap_data['known_skills']:
                print(f"   ✓ {skill['name'].title()}")
                print(f"     {skill['contribution']}")
        elif roadmap_data.get('is_beginner'):
            print(f"\n🆕  STARTING FRESH: No prior skills - complete beginner path")
        
        # Roadmap
        print(f"\n{'='*70}")
        print(f"📋  YOUR PERSONALIZED LEARNING PATH")
        print(f"{'='*70}")
        
        for step in roadmap_data['roadmap']:
            order = step.get('order', '?')
            skill = step['skill'].title()
            level = step['level']
            duration = step.get('duration_weeks', 2)
            
            print(f"\n{'─'*50}")
            print(f"📌  STEP {order}: {skill}")
            print(f"{'─'*50}")
            print(f"   📈 Level: {level.upper()}")
            print(f"   ⏱️  Duration: {duration} weeks")
            print(f"   🎯 Category: {step.get('category', 'general').upper()}")
            
            if step.get('ai_explanation'):
                print(f"   💡 {step['ai_explanation']}")
            
            if step.get('learning_strategy'):
                print(f"   📖 Strategy: {step['learning_strategy']}")
            
            if step.get('project_ideas'):
                print(f"   🛠️  Projects:")
                for idea in step['project_ideas'][:2]:
                    print(f"      • {idea}")
            
            if step.get('warning'):
                print(f"   ⚠️  Note: {step['warning']}")
        
        # Timeline
        print(f"\n{'='*70}")
        print(f"📅  ESTIMATED TOTAL TIME: {roadmap_data['total_duration_weeks']} weeks")
        print(f"📊  JOB READINESS: {roadmap_data.get('job_readiness_score', 0)}%")
        
        # Milestones
        if roadmap_data.get('milestones'):
            print(f"\n🏆  KEY MILESTONES:")
            for milestone in roadmap_data['milestones']:
                print(f"   ✨ After Step {milestone['after_step']}: {milestone['milestone']}")
                print(f"      {milestone['achievement']}")
        
        print(f"\n{'='*70}")
        print("💪  YOU'VE GOT THIS! CONSISTENCY IS KEY TO SUCCESS!")
        print(f"{'='*70}\n")

# =========================
# MAIN APPLICATION WITH DIALOGUE SUPPORT
# =========================
class CareerRoadmapAI:
    """Main application class with dialogue management"""
    
    def __init__(self, skills_path: str, train_path: str):
        self.models = ModelManager()
        self.skills_df = self._load_data(skills_path)
        available_careers = self.skills_df['subject'].unique().tolist()
        
        self.dialogue = DialogueManager(available_careers)
        self.skill_extractor = SkillExtractor(self.models, self.skills_df['skill'].unique().tolist())
        self.roadmap_generator = RoadmapGenerator(self.models, self.skills_df)
        self.ui = ModernUI()
    
    def _load_data(self, skills_path: str) -> pd.DataFrame:
        """Load and validate data"""
        try:
            df = pd.read_csv(skills_path)
            
            required_cols = ['subject', 'skill', 'level', 'duration_weeks']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'duration_weeks':
                        df[col] = 2
                    elif col == 'level':
                        df[col] = 'intermediate'
                    else:
                        raise ValueError(f"Missing required column: {col}")
            
            df = df.fillna({
                'prerequisite': 'none',
                'duration_weeks': 2,
                'level': 'intermediate'
            })
            
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                df[col] = df[col].astype(str).str.lower().str.strip()
            
            return df
            
        except FileNotFoundError:
            print("📝 Skills data not found. Creating demo dataset...")
            return self._create_demo_data()
    
    def _create_demo_data(self) -> pd.DataFrame:
        """Create comprehensive demo dataset"""
        demo_data = {
            'subject': [
                'ai engineer', 'ai engineer', 'ai engineer', 'ai engineer',
                'ai engineer', 'ai engineer', 'ai engineer', 'ai engineer',
                'data scientist', 'data scientist', 'data scientist', 'data scientist',
                'data scientist', 'data scientist',
                'frontend developer', 'frontend developer', 'frontend developer',
                'frontend developer', 'frontend developer',
                'cybersecurity', 'cybersecurity', 'cybersecurity', 'cybersecurity',
                'cybersecurity'
            ],
            'skill': [
                'python', 'numpy', 'pandas', 'machine learning',
                'deep learning', 'natural language processing', 'computer vision',
                'reinforcement learning',
                'python', 'sql', 'statistics', 'machine learning',
                'data visualization', 'big data',
                'html', 'css', 'javascript', 'react', 'typescript',
                'networking', 'linux', 'python', 'ethical hacking', 'cryptography'
            ],
            'level': [
                'beginner', 'beginner', 'beginner', 'intermediate',
                'advanced', 'advanced', 'advanced', 'expert',
                'beginner', 'beginner', 'intermediate', 'intermediate',
                'intermediate', 'advanced',
                'beginner', 'beginner', 'intermediate', 'intermediate', 'advanced',
                'beginner', 'beginner', 'intermediate', 'advanced', 'advanced'
            ],
            'duration_weeks': [
                4, 2, 2, 6, 8, 6, 6, 8,
                4, 2, 3, 6, 3, 6,
                2, 2, 4, 6, 4,
                3, 3, 4, 8, 6
            ],
            'prerequisite': [
                'none', 'python', 'python', 'python,numpy',
                'machine learning', 'python,deep learning', 'python,deep learning',
                'deep learning,machine learning',
                'none', 'none', 'python', 'python,sql,statistics',
                'python,pandas', 'python,sql',
                'none', 'none', 'html,css', 'javascript', 'javascript,react',
                'none', 'networking', 'python,linux', 'networking,linux', 'networking,python'
            ]
        }
        return pd.DataFrame(demo_data)
    
    def _extract_career(self, text: str) -> Optional[str]:
        """Extract career from user input"""
        text_lower = text.lower()
        
        career_patterns = {
            "ai engineer": ["ai engineer", "ai engineering", "ml engineer", "machine learning engineer"],
            "data scientist": ["data scientist", "data science"],
            "frontend developer": ["frontend", "front end", "frontend developer"],
            "cybersecurity": ["cybersecurity", "cyber security", "security engineer", "ethical hacker"],
            "data engineer": ["data engineer", "data engineering"],
            "devops engineer": ["devops", "devops engineer"],
            "mobile developer": ["mobile developer", "mobile dev", "ios", "android"],
            "game developer": ["game developer", "game dev", "game programming"]
        }
        
        for career, patterns in career_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return career
        
        career_labels = list(career_patterns.keys())
        try:
            result = self.models.intent_classifier(text, career_labels)
            if result['scores'][0] > 0.5:
                return result['labels'][0]
        except:
            pass
        
        return None
    
    def run(self):
        """Main application loop with dialogue support"""
        self.ui.display_welcome()
        
        print(f"✅ Loaded {len(self.skills_df)} skills across {self.skills_df['subject'].nunique()} career paths")
        print(f"📋 Available careers: {', '.join(self.skills_df['subject'].unique())}")
        print("\n" + "="*70)
        print("💬  Tell me about your career goals!")
        print("    Examples:")
        print("    • 'I want to become an AI Engineer'")
        print("    • 'Data Scientist with Python and SQL skills'")
        print("    • 'Frontend Developer starting from scratch'\n")
        
        while True:
            # Get initial user input
            user_input = input("You: ").strip()
            
            if not user_input:
                print("⚠️  Please tell me about your career goals!")
                continue
            
            # Extract career and skills
            detected_career = self._extract_career(user_input)
            detected_skills = self.skill_extractor.extract(user_input)
            
            # Process through dialogue manager
            result = self.dialogue.process_input(user_input, detected_career, detected_skills)
            
            action = result.get('action')
            
            # Handle different actions
            if action == 'exit':
                break
                
            elif action == 'restart':
                print("\n🔄 Starting fresh! Tell me about your career goals:\n")
                continue
                
            elif action == 'ask_for_skills':
                # KEY FIX: When only career is detected, ask for skills
                print(f"\n🎯 Career detected: {result['career'].upper()}")
                print(f"\n{result['message']}")
                skills_input = input("Skills: ").strip()
                
                # Process skills response
                extracted = self.skill_extractor.extract(skills_input)
                skills_result = self.dialogue.process_input(skills_input, None, extracted)
                
                if skills_result['action'] == 'confirm_no_skills':
                    print(f"\n{skills_result['message']}")
                    # Generate roadmap immediately for beginners
                    self._generate_and_display()
                    continue
                elif skills_result['action'] in ['confirm_skills', 'ask_skills_again']:
                    print(f"\n{skills_result['message']}")
                    confirm = input("Is this correct? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self._generate_and_display()
                        continue
                    else:
                        print("\nLet's try again. Tell me about your skills:")
                        continue
                
            elif action == 'ask_for_career':
                print(f"\n{result['message']}")
                career_input = input("Career: ").strip()
                new_career = self._extract_career(career_input)
                career_result = self.dialogue.process_input(career_input, new_career, [])
                
                if career_result['action'] == 'ask_for_skills':
                    print(f"\n{career_result['message']}")
                    skills_input = input("Skills: ").strip()
                    extracted = self.skill_extractor.extract(skills_input)
                    self.dialogue.process_input(skills_input, None, extracted)
                    self._generate_and_display()
                    continue
            
            elif action == 'ask_everything':
                print(f"\n{result['message']}")
                continue
                
            elif action in ['confirm_career_and_skills', 'confirm_skills', 'confirm_no_skills']:
                print(f"\n{result['message']}")
                confirm = input("Is this correct? (y/n): ").strip().lower()
                if confirm == 'y':
                    self._generate_and_display()
                    continue
                else:
                    self.dialogue.reset()
                    print("\n🔄 Okay, let's start over. Tell me about your goals:\n")
                    continue
            
            # Ask if user wants to continue
            if input("\n🔄 Generate another roadmap? (y/n): ").strip().lower() != 'y':
                break
        
        print("\n👋 Good luck on your learning journey!")
        print("💡 Remember: The best time to start is NOW!")
    
    def _generate_and_display(self):
        """Generate roadmap and display it"""
        print("\n📝 Generating your personalized roadmap...")
        print("   This uses AI to create an optimal learning path...\n")
        
        roadmap = self.roadmap_generator.generate_roadmap(
            self.dialogue.detected_career,
            self.dialogue.detected_skills,
            self.dialogue.experience_level
        )
        
        self.ui.display_roadmap(roadmap)
        self.dialogue.reset()

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    SKILLS_PATH = r"/content/SkillsRoadmap.csv"
    TRAIN_PATH = r"/content/TrainingData.csv"
    
    app = CareerRoadmapAI(SKILLS_PATH, TRAIN_PATH)
    app.run()