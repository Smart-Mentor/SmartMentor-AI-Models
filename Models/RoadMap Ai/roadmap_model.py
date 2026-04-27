import pandas as pd
import re
import random
import json
import os
import pickle
import ast
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
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
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
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
# ENHANCED TRANSFORMER-BASED SKILL EXTRACTOR WITH CONTEXT UNDERSTANDING
# =========================
class TransformerSkillExtractor:
    """Advanced skill extraction using BERT and transformers with context understanding"""
    
    def __init__(self):
        self.console = Console()
        self.model = None
        self.skill_embeddings = {}
        self.skill_synonyms = self._load_synonyms()
        self.domain_keywords = self._load_domain_keywords()
        self.context_patterns = self._load_context_patterns()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            with self.console.status("[bold green]Loading AI model (this may take a moment)...") as status:
                # Use a lightweight but effective model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.console.print("[green]✓ AI model loaded successfully![/green]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Could not load transformer model: {e}[/yellow]")
            self.console.print("[yellow]Falling back to enhanced basic extraction...[/yellow]")
            self.model = None
    
    def _load_synonyms(self) -> Dict:
        """Load comprehensive skill synonyms and related terms"""
        return {
            'python': ['py', 'python3', 'python programming', 'python development', 'cpython', 'anaconda'],
            'javascript': ['js', 'ecmascript', 'javascript programming', 'vanilla js', 'es6', 'node.js'],
            'machine learning': ['ml', 'machine learning ai', 'predictive modeling', 'statistical learning', 'ml algorithms', 'ai/ml'],
            'deep learning': ['dl', 'neural networks', 'deep neural networks', 'cnn', 'rnn', 'transformer networks'],
            'sql': ['structured query language', 'postgresql', 'mysql', 'sqlite', 'database query', 'pl/sql'],
            'react': ['reactjs', 'react.js', 'react framework', 'react native', 'next.js'],
            'aws': ['amazon web services', 'ec2', 's3', 'lambda', 'cloud computing aws'],
            'docker': ['containerization', 'docker containers', 'docker engine', 'kubernetes', 'devops tools'],
            'java': ['java8', 'java11', 'core java', 'spring boot', 'java enterprise'],
            'c++': ['cpp', 'c plus plus', 'cpp programming', 'c++11', 'c++14', 'c++17'],
            'html': ['html5', 'html/css', 'hypertext markup language', 'html dom'],
            'css': ['css3', 'stylesheet', 'flexbox', 'grid layout', 'css frameworks'],
            'data science': ['data analytics', 'data mining', 'big data', 'analytics', 'data visualization'],
            'cloud computing': ['cloud', 'aws', 'azure', 'gcp', 'cloud architecture'],
            'devops': ['ci/cd', 'jenkins', 'gitlab', 'automation', 'infrastructure as code']
        }
    
    def _load_domain_keywords(self) -> Dict:
        """Load domain-specific keywords for context understanding"""
        return {
            'web development': ['frontend', 'backend', 'fullstack', 'website', 'web app', 'api', 'rest', 'html', 'css', 'javascript', 'react'],
            'data science': ['analytics', 'statistics', 'visualization', 'prediction', 'modeling', 'insights'],
            'machine learning': ['algorithm', 'model', 'training', 'prediction', 'classification', 'regression', 'ai', 'artificial intelligence'],
            'software engineering': ['development', 'coding', 'programming', 'architecture', 'design patterns'],
            'cloud': ['deployment', 'scalability', 'infrastructure', 'serverless', 'container']
        }
    
    def _load_context_patterns(self) -> List[Dict]:
        """Load patterns for understanding skill context and corrections"""
        return [
            {
                'pattern': r'(?:i\s+said|i\s+meant|actually|correction|not\s+)\s*(\w+)\s+(?:i\s+meant|meaning|actually|want|mean)\s+(\w+(?:\s+\w+)*)',
                'weight': 1.5,
                'description': 'Correction patterns'
            },
            {
                'pattern': r'(?:i\s+want\s+to\s+learn|interested\s+in|focus\s+on)\s+(\w+(?:\s+\w+)*)',
                'weight': 1.3,
                'description': 'Learning intention'
            },
            {
                'pattern': r'(?:my\s+goal\s+is|aiming\s+to\s+become|career\s+goal\s+is)\s+(\w+(?:\s+\w+)*)',
                'weight': 1.4,
                'description': 'Career goals'
            }
        ]
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using transformer model"""
        if self.model is None:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            return None
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.model is None:
            # Fallback to simple string matching
            return 1.0 if text1.lower() in text2.lower() or text2.lower() in text1.lower() else 0.0
        
        try:
            emb1 = self.model.encode(text1, convert_to_numpy=True)
            emb2 = self.model.encode(text2, convert_to_numpy=True)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def understand_skill_context(self, text: str) -> Dict:
        """Understand the context and possible corrections in user input"""
        text_lower = text.lower()
        context_info = {
            'corrections': [],
            'intentions': [],
            'primary_focus': None,
            'confidence': 0.0,
            'domain': 'general'
        }
        
        # Detect which domain the user is interested in
        web_terms = ['web', 'website', 'frontend', 'backend', 'fullstack', 'html', 'css', 'javascript', 'react', 'angular', 'vue']
        ml_terms = ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural networks', 'data science']
        
        # Count domain-specific terms
        web_count = sum(1 for term in web_terms if term in text_lower)
        ml_count = sum(1 for term in ml_terms if term in text_lower)
        
        # Determine primary domain
        if ml_count > web_count:
            context_info['domain'] = 'machine_learning'
        elif web_count > ml_count:
            context_info['domain'] = 'web_development'
        else:
            context_info['domain'] = 'general'
        
        # Check for correction patterns (e.g., "I said HTML but meant Machine Learning")
        correction_pattern = r'(?:i\s+said|i\s+meant|actually|correction|not)\s+(\w+)\s+(?:but|i\s+meant|meaning|actually|want|mean)\s+(\w+(?:\s+\w+)*)'
        correction_matches = re.findall(correction_pattern, text_lower, re.IGNORECASE)
        
        for match in correction_matches:
            if len(match) >= 2:
                context_info['corrections'].append({
                    'original': match[0],
                    'corrected': match[1],
                    'type': 'explicit_correction'
                })
                # If correction mentions ML, update domain
                if 'machine learning' in match[1] or 'ml' in match[1] or 'ai' in match[1]:
                    context_info['domain'] = 'machine_learning'
        
        # Check for intention patterns
        for pattern_info in self.context_patterns:
            matches = re.findall(pattern_info['pattern'], text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = ' '.join(match)
                else:
                    match_text = match
                context_info['intentions'].append({
                    'text': match_text,
                    'weight': pattern_info['weight'],
                    'type': pattern_info['description']
                })
                
                # Update domain based on intention
                if any(term in match_text for term in ml_terms):
                    context_info['domain'] = 'machine_learning'
                elif any(term in match_text for term in web_terms):
                    context_info['domain'] = 'web_development'
        
        return context_info
    
    def extract_skills_advanced(self, text: str, all_skills: List[str], threshold: float = 0.6) -> Tuple[List[str], Dict]:
        """Advanced skill extraction using transformer embeddings with context understanding"""
        if not text or text.lower() in ['none', 'no skills', 'beginner', 'nothing', '']:
            return [], {'method': 'empty', 'confidence': 1.0}
        
        text = str(text).lower()
        detected = []
        confidence_scores = {}
        
        # First, understand context and possible corrections
        context = self.understand_skill_context(text)
        
        # Apply corrections if any
        processed_text = text
        for correction in context['corrections']:
            if correction['original'] in processed_text:
                # Replace the incorrect term with corrected one
                processed_text = processed_text.replace(correction['original'], correction['corrected'])
                console.print(f"[dim]✓ Understood correction: {correction['original']} → {correction['corrected']}[/dim]")
        
        # Method 1: Direct matching with synonyms
        for skill in all_skills:
            skill_str = str(skill).lower()
            
            # Direct match
            if skill_str in processed_text:
                # Check if this is HTML and context is ML - then skip HTML
                if skill_str == 'html' and context['domain'] == 'machine_learning':
                    console.print(f"[dim]ℹ️ Detected 'html' but context suggests Machine Learning, skipping HTML[/dim]")
                    continue
                detected.append(skill_str)
                confidence_scores[skill_str] = 1.0
                continue
            
            # Synonym matching
            for key, synonyms in self.skill_synonyms.items():
                if key in skill_str or skill_str in key:
                    for syn in synonyms:
                        if syn in processed_text:
                            # Special handling for HTML in ML context
                            if skill_str == 'html' and context['domain'] == 'machine_learning':
                                continue
                            detected.append(skill_str)
                            confidence_scores[skill_str] = 0.9
                            break
                    if skill_str in detected:
                        break
        
        # Method 2: Semantic similarity using transformer (only if needed)
        if self.model is not None and len(detected) < 5:
            text_embedding = self.get_text_embedding(processed_text)
            
            if text_embedding is not None:
                for skill in all_skills:
                    skill_str = str(skill).lower()
                    if skill_str not in detected and skill_str != 'html':  # Skip HTML semantic matching
                        # Get skill embedding (cache for performance)
                        if skill_str not in self.skill_embeddings:
                            self.skill_embeddings[skill_str] = self.get_text_embedding(skill_str)
                        
                        skill_embedding = self.skill_embeddings[skill_str]
                        if skill_embedding is not None:
                            similarity = cosine_similarity([text_embedding], [skill_embedding])[0][0]
                            
                            # Boost similarity if there's context indicating this skill
                            if context['intentions']:
                                similarity *= 1.1
                            
                            if similarity > threshold:
                                detected.append(skill_str)
                                confidence_scores[skill_str] = float(similarity)
        
        # Method 3: Contextual extraction based on domain
        if context['domain'] == 'web_development' and 'html' in processed_text and 'html' not in detected:
            # User is in web development context, so HTML is valid
            detected.append('html')
            confidence_scores['html'] = 0.9
            console.print(f"[green]✓ Detected 'html' in web development context[/green]")
        
        # Remove duplicates and return
        detected = list(set(detected))
        
        # Sort by confidence if available
        if confidence_scores:
            detected.sort(key=lambda x: confidence_scores.get(x, 0), reverse=True)
        
        return detected[:15], {'method': 'transformer', 'confidence': np.mean(list(confidence_scores.values())) if confidence_scores else 0.5}
    
    def extract_skills(self, text: str, all_skills: List[str]) -> List[str]:
        """Main extraction method with detailed analysis"""
        skills, metadata = self.extract_skills_advanced(text, all_skills)
        return skills

# =========================
# ENHANCED CAREER UNDERSTANDER WITH CONTEXT
# =========================
class CareerUnderstander:
    """Understands user career goals using transformer models with mandatory input handling"""
    
    def __init__(self):
        self.model = None
        self.career_embeddings = {}
        self.career_categories = self._load_career_categories()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize transformer model for career understanding"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[green]✓ Career understanding AI ready![/green]")
        except:
            self.model = None
            console.print("[yellow]⚠️ Using basic career matching[/yellow]")
    
    def _load_career_categories(self) -> Dict:
        """Load career categories with synonyms and related terms"""
        return {
            'data scientist': {
                'keywords': ['data science', 'machine learning', 'analytics', 'data mining', 'big data', 'ai', 'statistics', 'data analysis'],
                'synonyms': ['data analyst', 'data engineer', 'ml engineer', 'data architect', 'business analyst']
            },
            'web developer': {
                'keywords': ['web dev', 'frontend', 'backend', 'fullstack', 'website', 'web app', 'javascript', 'react', 'angular', 'vue', 'html', 'css'],
                'synonyms': ['frontend developer', 'backend developer', 'full stack developer', 'web programmer']
            },
            'software engineer': {
                'keywords': ['software dev', 'programming', 'coding', 'application development', 'systems', 'software development'],
                'synonyms': ['software developer', 'programmer', 'software architect', 'application developer']
            },
            'devops engineer': {
                'keywords': ['devops', 'ci/cd', 'automation', 'deployment', 'infrastructure', 'cloud operations', 'sre'],
                'synonyms': ['site reliability engineer', 'platform engineer', 'cloud engineer', 'infrastructure engineer']
            },
            'cloud architect': {
                'keywords': ['cloud', 'aws', 'azure', 'gcp', 'cloud infrastructure', 'scalability', 'cloud computing'],
                'synonyms': ['cloud engineer', 'cloud consultant', 'cloud solutions architect']
            },
            'ai/ml engineer': {
                'keywords': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision'],
                'synonyms': ['ai researcher', 'ml engineer', 'deep learning engineer', 'ai specialist']
            },
            'machine learning engineer': {
                'keywords': ['ml', 'machine learning', 'model deployment', 'mlops', 'feature engineering', 'model training', 'ai'],
                'synonyms': ['ml engineer', 'machine learning specialist', 'ai/ml engineer']
            }
        }
    
    def get_career_input(self, available_careers: List[str] = None) -> Tuple[str, float, Dict]:
        """Get and validate career input from user - ensures they don't skip it"""
        console.print("\n[cyan]🎯 Tell me about your career goal[/cyan]")
        console.print("[dim]Example: 'I want to become a data scientist' or 'Machine Learning Engineer'[/dim]")
        console.print("[bold green]💡 Be as specific as you like - I'll understand![/bold green]")
        console.print("[bold yellow]⚠️ This is required to create your personalized roadmap[/bold yellow]\n")
        
        while True:
            career_input = Prompt.ask("[yellow]What career are you aiming for?[/yellow]").strip()
            
            # Check if input is empty or just whitespace
            if not career_input or career_input.lower() in ['', 'none', 'skip', 'pass', 'idk', "i don't know", 'not sure']:
                console.print("[red]❌ Career goal is required to create your learning path![/red]")
                console.print("[cyan]💡 Popular careers: Data Scientist, Web Developer, Machine Learning Engineer, Software Engineer[/cyan]")
                continue
            
            # Process the career input
            matched_career, confidence, metadata = self.understand_career_goal(career_input, available_careers)
            
            if confidence > 0.5:
                console.print(f"\n[green]✓ I understand you want to become: {matched_career.title()}[/green]")
                
                # Ask for confirmation
                confirm = Confirm.ask(f"Is this correct?", default=True)
                if confirm:
                    return matched_career, confidence, metadata
                else:
                    console.print("[yellow]Let me try again. Please rephrase your career goal.[/yellow]")
                    continue
            else:
                console.print(f"[yellow]⚠️ I'm not sure I understood '{career_input}'.[/yellow]")
                console.print("[cyan]💡 Try one of these examples:[/cyan]")
                console.print("   • Data Scientist (for data analysis and ML)")
                console.print("   • Web Developer (for building websites and applications)")
                console.print("   • Machine Learning Engineer (for AI and model deployment)")
                console.print("   • Software Engineer (for general software development)")
                continue
    
    def understand_career_goal(self, user_input: str, available_careers: List[str] = None) -> Tuple[str, float, Dict]:
        """Understand and match career goal using transformer embeddings"""
        user_input = str(user_input).lower().strip()
        
        # Special handling for beginners or unclear input
        if user_input in ['', 'none', 'not sure', 'undecided', '?', 'help', 'idk', "i don't know"]:
            return "beginner_exploration", 0.5, {"message": "Exploring career options"}
        
        results = {}
        
        # Method 1: Direct matching with available careers
        if available_careers:
            for career in available_careers:
                career_str = str(career).lower()
                similarity = self._calculate_similarity(user_input, career_str)
                results[career_str] = similarity
        
        # Method 2: Match with career categories
        for category, data in self.career_categories.items():
            # Check category name
            similarity = self._calculate_similarity(user_input, category)
            
            # Check keywords
            for keyword in data['keywords']:
                sim = self._calculate_similarity(user_input, keyword)
                similarity = max(similarity, sim * 0.9)
            
            # Check synonyms
            for synonym in data['synonyms']:
                sim = self._calculate_similarity(user_input, synonym)
                similarity = max(similarity, sim * 0.85)
            
            # Special handling for ML/AI related careers
            if 'machine learning' in user_input or 'ml' in user_input or 'ai' in user_input:
                if 'machine learning' in category or 'ai' in category:
                    similarity = max(similarity, 0.9)  # High confidence for ML careers
            
            # Special handling for web development with HTML
            if 'html' in user_input or 'css' in user_input or 'javascript' in user_input:
                if 'web developer' in category:
                    similarity = max(similarity, 0.85)
            
            results[category] = similarity
        
        # Get best match
        if results:
            best_match = max(results.items(), key=lambda x: x[1])
            career, confidence = best_match
            
            # Boost confidence for ML-related inputs
            if any(term in user_input for term in ['machine learning', 'ml', 'ai', 'artificial intelligence']):
                if 'machine learning' in career or 'ai' in career:
                    confidence = min(confidence * 1.2, 1.0)
                    career = 'machine learning engineer'  # Standardize the career name
            
            # Extract specific domain from user input
            domain = self._extract_domain(user_input)
            
            return career, confidence, {
                'original_input': user_input,
                'domain': domain,
                'all_matches': {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]},
                'method': 'transformer' if self.model else 'basic'
            }
        
        # If no matches found, return the user input as is with low confidence
        return user_input.split()[-1] if user_input.split() else user_input, 0.3, {'method': 'fallback', 'confidence': 0.3}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.model is not None:
            try:
                emb1 = self.model.encode(text1, convert_to_numpy=True)
                emb2 = self.model.encode(text2, convert_to_numpy=True)
                return float(cosine_similarity([emb1], [emb2])[0][0])
            except:
                pass
        
        # Fallback to token matching
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        # Add substring matching bonus
        substring_bonus = 0
        if text2 in text1 or text1 in text2:
            substring_bonus = 0.3
        
        return (intersection / union) + substring_bonus if union > 0 else 0.0
    
    def _extract_domain(self, text: str) -> str:
        """Extract specific domain from user input"""
        domains = {
            'web': ['web', 'frontend', 'backend', 'fullstack', 'website', 'react', 'angular', 'vue', 'html', 'css'],
            'data': ['data', 'analytics', 'statistics', 'visualization', 'database'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'nlp', 'ml'],
            'cloud': ['cloud', 'aws', 'azure', 'gcp', 'deployment'],
            'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter']
        }
        
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in text:
                    return domain
        
        return "general"

# =========================
# ENHANCED KNOWLEDGE GRAPH WITH EMBEDDINGS
# =========================
class EnhancedKnowledgeGraph:
    """Knowledge graph with transformer embeddings for better skill relationships"""
    
    def __init__(self):
        self.skills = {}
        self.prerequisites = defaultdict(list)
        self.skill_embeddings = {}
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model for skill embeddings"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None
    
    def add_skill(self, skill: str, difficulty: int = 1, description: str = ""):
        """Add skill with embedding"""
        skill_str = str(skill)
        self.skills[skill_str] = {
            'difficulty': difficulty,
            'description': description,
            'embedding': self._get_embedding(f"{skill_str} {description}")
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self.model is None:
            return None
        
        if text in self.skill_embeddings:
            return self.skill_embeddings[text]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.skill_embeddings[text] = embedding
            return embedding
        except:
            return None
    
    def add_prerequisite(self, skill: str, prerequisite: str):
        """Add prerequisite relationship"""
        skill_str = str(skill)
        prereq_str = str(prerequisite)
        self.prerequisites[skill_str].append(prereq_str)
    
    def find_related_skills(self, skill: str, all_skills: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantically related skills using embeddings"""
        if self.model is None or skill not in self.skills:
            return []
        
        skill_embedding = self.skills.get(skill, {}).get('embedding')
        if skill_embedding is None:
            return []
        
        similarities = []
        for other_skill in all_skills:
            other_skill_str = str(other_skill)
            if other_skill_str != skill and other_skill_str in self.skills:
                other_embedding = self.skills[other_skill_str].get('embedding')
                if other_embedding is not None:
                    similarity = cosine_similarity([skill_embedding], [other_embedding])[0][0]
                    similarities.append((other_skill_str, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_smart_roadmap(self, career_goal: str, all_skills_df, user_skills: List[str] = None) -> List[str]:
        """Generate intelligent roadmap using embeddings and prerequisites"""
        career_goal_str = str(career_goal)
        
        # Get career skills
        career_skills = all_skills_df[all_skills_df['subject'] == career_goal_str]
        
        if career_skills.empty:
            return self._get_fallback_roadmap(career_goal_str)
        
        # Extract unique skills
        unique_skills = career_skills['skill'].unique().tolist()
        unique_skills = [str(s) for s in unique_skills if s and str(s).lower() not in ['nan', 'none', 'null', '']]
        
        # Remove skills user already has
        if user_skills:
            user_skills_set = set(str(s).lower() for s in user_skills)
            unique_skills = [s for s in unique_skills if s.lower() not in user_skills_set]
        
        # Sort by prerequisites (skills that are prerequisites should come first)
        skill_order = []
        for skill in unique_skills:
            prereqs = self.prerequisites.get(skill, [])
            prereq_count = len([p for p in prereqs if p in unique_skills])
            skill_order.append((skill, prereq_count))
        
        # Sort: fewer prerequisites come first (foundation skills)
        skill_order.sort(key=lambda x: x[1])
        
        return [skill for skill, _ in skill_order][:20]
    
    def _get_fallback_roadmap(self, career_goal: str) -> List[str]:
        """Enhanced fallback roadmaps with better understanding"""
        career_lower = career_goal.lower()
        
        fallback_roadmaps = {
            'data scientist': [
                'python programming basics', 'mathematics for data science', 'statistics fundamentals',
                'data cleaning and preprocessing', 'exploratory data analysis', 'data visualization',
                'machine learning algorithms', 'model evaluation and selection', 'feature engineering',
                'deep learning basics', 'sql for data science', 'big data fundamentals'
            ],
            'machine learning engineer': [
                'python programming', 'mathematics and linear algebra', 'statistics and probability',
                'data preprocessing', 'machine learning algorithms', 'model evaluation and tuning',
                'deep learning fundamentals', 'mlops and model deployment', 'feature engineering',
                'sql and database management', 'cloud platforms (aws/gcp)', 'version control with git'
            ],
            'web developer': [
                'html5 fundamentals', 'css3 and styling', 'responsive web design', 'javascript essentials',
                'dom manipulation', 'async javascript', 'version control with git', 'frontend framework (react/vue)',
                'backend development basics', 'database integration', 'restful apis', 'deployment and hosting'
            ],
            'software engineer': [
                'programming fundamentals', 'data structures and algorithms', 'object-oriented programming',
                'database design', 'software architecture', 'design patterns', 'testing and debugging',
                'version control', 'system design', 'agile methodologies', 'api development', 'security basics'
            ],
            'devops engineer': [
                'linux system administration', 'networking fundamentals', 'scripting (bash/python)',
                'version control with git', 'ci/cd pipelines', 'containerization with docker',
                'orchestration (kubernetes)', 'infrastructure as code', 'cloud platforms (aws/azure)',
                'monitoring and logging', 'security best practices', 'site reliability engineering'
            ]
        }
        
        # Find best matching roadmap
        for key, roadmap in fallback_roadmaps.items():
            if key in career_lower or career_lower in key:
                return roadmap
        
        # Generate dynamic roadmap based on career name
        return [
            f"Introduction to {career_goal}",
            f"Core Fundamentals of {career_goal}",
            f"Essential Tools for {career_goal}",
            f"Practical {career_goal} Skills",
            f"Advanced {career_goal} Concepts",
            f"{career_goal} Project Development",
            f"{career_goal} Best Practices",
            f"{career_goal} Career Preparation"
        ]

# =========================
# SIMPLIFIED UI COMPONENTS
# =========================
class SimpleUI:
    """Simplified terminal UI without complex visualizations"""
    @staticmethod
    def show_user_info(user_profile: UserProfile):
        """Display user information"""
        console.print("\n[bold cyan]📋 Your Learning Profile[/bold cyan]")
        console.print(f"  🎯 Goal: {user_profile.career_goal}")
        console.print(f"  ⭐ Level: {user_profile.experience_level.upper()}")
        console.print(f"  📚 Style: {user_profile.learning_style}")
        console.print(f"  ⏰ Hours/Week: {user_profile.available_hours_per_week}")
        
        if user_profile.skills:
            console.print(f"  🔧 Current Skills: {', '.join(user_profile.skills[:10])}")
        else:
            console.print(f"  🔧 Current Skills: [yellow]Beginning your learning journey![/yellow]")
    
    @staticmethod
    def show_learning_path(learning_path: LearningPath, user_profile: UserProfile):
        """Display learning path in a clean, readable format"""
        console.print("\n" + "="*60)
        console.print(f"[bold cyan]📚 {learning_path.title}[/bold cyan]")
        console.print("="*60)
        
        # Path summary
        console.print(f"\n[bold green]📊 Path Summary:[/bold green]")
        console.print(f"  • Type: {learning_path.track_type}")
        console.print(f"  • Difficulty: {learning_path.difficulty.upper()}")
        console.print(f"  • Total Skills: {len(learning_path.skills)}")
        console.print(f"  • Estimated Hours: {learning_path.estimated_hours}")
        console.print(f"  • Weeks to Complete: {learning_path.estimated_hours // max(user_profile.available_hours_per_week, 1)}")
        
        # Show learning roadmap
        console.print(f"\n[bold yellow]🗺️ Your Learning Roadmap:[/bold yellow]")
        console.print("")
        
        for i, skill in enumerate(learning_path.skills, 1):
            if i <= 3:
                icon = "🌟"
            elif i <= 6:
                icon = "📘"
            else:
                icon = "📖"
            
            console.print(f"  {icon} [bold white]Step {i}:[/bold white] [cyan]{skill}[/cyan]")
            hours_per_skill = learning_path.estimated_hours // max(len(learning_path.skills), 1)
            console.print(f"     [dim]⏱️ Estimated time: {hours_per_skill} hours[/dim]")
            
            if i < len(learning_path.skills):
                console.print(f"     [dim]↓[/dim]")
        
        # Show milestones
        if learning_path.milestones:
            console.print(f"\n[bold magenta]🎯 Key Milestones:[/bold magenta]")
            for i, milestone in enumerate(learning_path.milestones[:5], 1):
                console.print(f"  {i}. {milestone['title']} - {milestone['hours']} hours")
        
        # Show projects
        if learning_path.projects:
            console.print(f"\n[bold green]💼 Practical Projects:[/bold green]")
            for project in learning_path.projects[:3]:
                console.print(f"  • {project['name']}")
                console.print(f"    [dim]{project['description']}[/dim]")
        
        # Show resources
        if learning_path.resources:
            console.print(f"\n[bold blue]📚 Recommended Resources:[/bold blue]")
            for resource in learning_path.resources[:3]:
                console.print(f"  • [{resource['type']}] {resource['title']}")
        
        console.print("\n" + "="*60)

# =========================
# MAIN APPLICATION WITH TRANSFORMERS
# =========================
class SmartMentorAI:
    """Main application with transformer-based understanding"""
    
    def __init__(self):
        self.ui = SimpleUI()
        self.skill_extractor = None
        self.career_understander = None
        self.graph = None
        self.current_user: Optional[UserProfile] = None
        self.all_skills = []
        self.careers = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize with transformer models"""
        with console.status("[bold green]Loading AI-powered mentor system...") as status:
            try:
                # Load CSV files
                skills_path = r"D:/Games/RoadMap Ai/skills_roadmap.csv"
                train_path = r"D:/Games/RoadMap Ai/training_data22.csv"
                
                if not os.path.exists(skills_path):
                    console.print(f"[red]❌ File not found: {skills_path}[/red]")
                    raise FileNotFoundError(f"Skills roadmap file not found")
                
                if not os.path.exists(train_path):
                    console.print(f"[red]❌ File not found: {train_path}[/red]")
                    raise FileNotFoundError(f"Training data file not found")
                
                # Load data with proper handling
                self.skills_df = pd.read_csv(skills_path)
                self.train_df = pd.read_csv(train_path)
                
                # Fill NaN values with empty strings
                self.skills_df = self.skills_df.fillna('')
                self.train_df = self.train_df.fillna('')
                
                # Normalize data
                for col in ['skill', 'prerequisite', 'subject']:
                    if col in self.skills_df.columns:
                        self.skills_df[col] = self.skills_df[col].astype(str)
                        self.skills_df[col] = self.skills_df[col].str.lower().str.strip()
                        self.skills_df[col] = self.skills_df[col].replace('nan', '')
                        self.skills_df[col] = self.skills_df[col].replace('', '')
                
                # Filter out empty skills
                self.skills_df = self.skills_df[self.skills_df['skill'] != '']
                self.all_skills = self.skills_df['skill'].unique().tolist()
                self.all_skills = [str(s) for s in self.all_skills if s and str(s).lower() not in ['nan', 'none', 'null', '']]
                
                self.careers = self.skills_df['subject'].unique().tolist()
                self.careers = [str(c) for c in self.careers if c and str(c).lower() not in ['nan', 'none', 'null', '']]
                
                console.print(f"[green]✓ Loaded {len(self.all_skills)} skills and {len(self.careers)} careers[/green]")
                
                # Initialize transformer-based components
                status.update("[bold cyan]Initializing AI models...[/bold cyan]")
                self.skill_extractor = TransformerSkillExtractor()
                self.career_understander = CareerUnderstander()
                self._build_knowledge_graph()
                
                status.update("[bold green]✓ AI Mentor Ready![/bold green]")
                time.sleep(1)
                
            except Exception as e:
                console.print(f"[red]❌ Initialization failed: {e}[/red]")
                import traceback
                traceback.print_exc()
                console.print("[yellow]Please check your data files and try again[/yellow]")
                raise
    
    def _build_knowledge_graph(self):
        """Build enhanced knowledge graph with embeddings"""
        self.graph = EnhancedKnowledgeGraph()
        
        # Add skills with descriptions
        for _, row in self.skills_df.iterrows():
            skill = str(row.get('skill', ''))
            description = str(row.get('description', '')) if 'description' in row else ''
            difficulty = 1
            
            if 'advanced' in skill or 'expert' in skill:
                difficulty = 3
            elif 'intermediate' in skill or 'professional' in skill:
                difficulty = 2
            
            if skill and skill not in ['nan', 'none', 'null', '']:
                self.graph.add_skill(skill, difficulty, description)
        
        # Add prerequisites
        for _, row in self.skills_df.iterrows():
            prereq = str(row.get('prerequisite', ''))
            skill = str(row.get('skill', ''))
            
            if (skill and prereq and 
                skill not in ['nan', 'none', 'null', ''] and 
                prereq not in ['nan', 'none', 'null', '']):
                self.graph.add_prerequisite(skill, prereq)
    
    def create_user_profile(self) -> UserProfile:
        """Create user profile with AI-powered understanding"""
        console.print("\n[bold cyan]✨ Let's create your AI-powered learning profile[/bold cyan]\n")
        
        name = Prompt.ask("[yellow]What's your name?[/yellow]")
        
        # Learning style with AI recommendation
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
        
        # Career goal with AI understanding - MANDATORY, CANNOT SKIP
        matched_career, confidence, career_metadata = self.career_understander.get_career_input(self.careers)
        
        # Skills input with AI extraction and context understanding
        console.print("\n[cyan]🔧 Tell me about your current skills[/cyan]")
        console.print("[dim]Example: 'I know Python, some JavaScript, and I'm learning React'[/dim]")
        console.print("[bold green]💡 Just describe your skills naturally - I'll understand![/bold green]")
        console.print("[dim]Note: 'HTML' will be recognized as a web development skill unless you specify ML context[/dim]")
        
        skills_input = Prompt.ask("Your skills", default="none")
        
        # Use AI to extract skills
        if skills_input and skills_input.lower() not in ['none', 'no skills', 'beginner', 'nothing', '']:
            with console.status("[cyan]Analyzing your skills with AI...[/cyan]"):
                skills = self.skill_extractor.extract_skills(skills_input, self.all_skills)
                
                if skills:
                    console.print(f"\n[green]✓ I detected these skills: {', '.join(skills[:8])}[/green]")
                    if len(skills) > 8:
                        console.print(f"[dim]... and {len(skills)-8} more[/dim]")
                else:
                    console.print("[yellow]⚠️ I couldn't detect specific skills. We'll start from beginner level![/yellow]")
                    skills = []
                    experience = "beginner"
        else:
            console.print("[green]✓ Great! We'll create a complete beginner roadmap for you![/green]")
            skills = []
            experience = "beginner"
        
        return UserProfile(
            user_id=hashlib.md5(name.encode()).hexdigest()[:8],
            skills=skills,
            career_goal=matched_career,
            learning_style=learning_style,
            pace="moderate",
            available_hours_per_week=int(hours),
            experience_level=experience
        )
    
    def generate_personalized_path(self, user_profile: UserProfile) -> Tuple[LearningPath, str]:
        """Generate personalized learning path with AI understanding"""
        
        # Check if user is a beginner
        if not user_profile.skills or len(user_profile.skills) == 0:
            console.print("\n[bold cyan]🧠 Creating your AI-powered beginner roadmap...[/bold cyan]")
            roadmap_skills = self.graph.get_smart_roadmap(
                user_profile.career_goal, 
                self.skills_df,
                user_profile.skills
            )
        else:
            console.print("\n[bold cyan]🧠 Analyzing your skills and creating personalized path...[/bold cyan]")
            roadmap_skills = self.graph.get_smart_roadmap(
                user_profile.career_goal,
                self.skills_df,
                user_profile.skills
            )
        
        # Ensure we have skills
        if not roadmap_skills:
            roadmap_skills = self.graph._get_fallback_roadmap(user_profile.career_goal)
        
        # Calculate estimated hours based on difficulty and user availability
        total_hours = len(roadmap_skills) * 30  # 30 hours per skill average
        weeks_to_complete = total_hours // max(user_profile.available_hours_per_week, 1)
        
        # Create resources list properly
        resources = []
        for skill in roadmap_skills[:3]:
            resources.append({"type": "Course", "title": f"Complete {skill}", "url": "#"})
            resources.append({"type": "Documentation", "title": f"{skill} Official Docs", "url": "#"})
            resources.append({"type": "Practice Platform", "title": f"Hands-on {skill} Exercises", "url": "#"})
        
        # Create projects list properly
        projects = []
        for skill in roadmap_skills[:3]:
            projects.append({"name": f"Build with {skill}", "description": f"Practical project using {skill}"})
        
        # Create quizzes list properly
        quizzes = []
        for skill in roadmap_skills[:2]:
            quizzes.append({"title": f"{skill} Assessment", "questions": 20, "passing_score": 70})
        
        # Create milestones list properly
        milestones = []
        for skill in roadmap_skills[:5]:
            milestones.append({"title": f"Master {skill}", "hours": 30, "completed": False})
        
        # Create learning path
        learning_path = LearningPath(
            title=f"AI-Powered Path to {user_profile.career_goal.title()}",
            track_type="Personalized Learning Path",
            skills=roadmap_skills[:15],
            estimated_hours=total_hours,
            difficulty=user_profile.experience_level,
            prerequisites=user_profile.skills[:5],
            milestones=milestones,
            resources=resources,
            projects=projects,
            quizzes=quizzes,
            created_at=datetime.now().isoformat()
        )
        
        return learning_path, "AI-Powered Path"
    
    def run(self):
        """Main application loop"""
        try:
            # Create user profile
            self.current_user = self.create_user_profile()
            
            while True:
                # Display user info
                self.ui.show_user_info(self.current_user)
                
                # Generate learning path
                learning_path, track = self.generate_personalized_path(self.current_user)
                
                # Display the roadmap
                self.ui.show_learning_path(learning_path, self.current_user)
                
                # Show AI insights
                console.print("\n[bold cyan]🤖 AI Insights:[/bold cyan]")
                console.print(f"[dim]• This path focuses on skills most relevant to {self.current_user.career_goal}[/dim]")
                console.print(f"[dim]• Estimated completion: {learning_path.estimated_hours // self.current_user.available_hours_per_week} weeks[/dim]")
                console.print(f"[dim]• Start with the first skill and practice {self.current_user.available_hours_per_week} hours/week[/dim]")
                
                # Ask for feedback
                console.print("\n[cyan]💬 Was this roadmap helpful?[/cyan]")
                feedback = Prompt.ask(
                    "Rate this path",
                    choices=["excellent", "good", "okay", "poor"],
                    default="good"
                )
                
                if feedback == "excellent":
                    console.print("[green]🎉 Great! I'm glad the AI recommendations were helpful![/green]")
                elif feedback == "poor":
                    console.print("[yellow]📝 Thanks for feedback! I'll learn and improve.[/yellow]")
                
                # Ask for continuation
                if not Confirm.ask("\n[cyan]🔄 Would you like to see another path?[/cyan]", default=False):
                    console.print("\n[bold green]🎉 Your AI mentor believes in you![/bold green]")
                    console.print("[dim]💡 Pro tip: Consistency is key - study a little every day![/dim]")
                    console.print(f"[dim]🚀 Start with: {learning_path.skills[0] if learning_path.skills else 'the first skill'}[/dim]")
                    break
                
                # Option to update skills
                if Confirm.ask("Have you learned any new skills?", default=False):
                    new_skills_input = Prompt.ask("Describe what you've learned", default="")
                    if new_skills_input and new_skills_input.lower() not in ['none', 'no']:
                        new_skills = self.skill_extractor.extract_skills(new_skills_input, self.all_skills)
                        if new_skills:
                            self.current_user.skills.extend(new_skills)
                            self.current_user.skills = list(set(self.current_user.skills))
                            console.print(f"[green]✓ Updated! You now have {len(self.current_user.skills)} skills[/green]")
                            
                            # Update experience level
                            if len(self.current_user.skills) > 5:
                                self.current_user.experience_level = "intermediate"
                                console.print("[green]✓ Great progress! You're now at Intermediate level![/green]")
                            elif len(self.current_user.skills) > 10:
                                self.current_user.experience_level = "advanced"
                                console.print("[green]🌟 Outstanding! You've reached Advanced level![/green]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Keep learning! Your AI mentor is always here![/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            import traceback
            traceback.print_exc()

# =========================
# RUN APPLICATION
# =========================
if __name__ == "__main__":
    try:
        console.print("[bold blue]🤖 AI-Powered Smart Mentor System[/bold blue]")
        console.print("[dim]Using advanced AI to understand your learning needs[/dim]")
        console.print("[dim]✨ The AI understands context - HTML is for web development unless specified otherwise![/dim]\n")
        
        mentor = SmartMentorAI()
        mentor.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        console.print("[yellow]Please ensure your CSV files exist at the correct paths[/yellow]")
        console.print("[dim]You can install required packages: pip install sentence-transformers scikit-learn pandas rich numpy torch[/dim]")
        input("Press Enter to exit...")