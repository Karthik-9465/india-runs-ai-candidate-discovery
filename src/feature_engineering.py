# src/feature_engineering.py
class CandidateFeatureExtractor:
    """Feature Engineering Layer that extracts rich structured features for every candidate."""
    
    def __init__(self, boilerplate_descriptions=None):
        # A set of boilerplate descriptions to detect filler profiles and compute career consistency
        self.boilerplate_descriptions = boilerplate_descriptions or set()
        
        # Skill groups mapping for feature engineering
        self.skill_keywords = {
            "ai": ["ai", "artificial intelligence", "cognitive", "deep learning", "neural network"],
            "ml": ["machine learning", "ml", "data science", "statistics", "statistical", "regression", "classification", "clustering"],
            "llm": ["llm", "large language model", "transformer", "bert", "gpt", "llama", "mistral", "fine-tuning", "lora", "qlora", "peft"],
            "rag": ["rag", "retrieval-augmented generation", "langchain", "llamaindex", "document ingestion", "chunking"],
            "retrieval": ["retrieval", "search", "information retrieval", "dense retrieval", "semantic search", "hybrid search", "bm25", "elasticsearch", "opensearch", "faiss"],
            "ranking": ["ranking", "learning-to-rank", "ltr", "xgboost", "lightgbm", "relevance", "recommendation", "collaborative filtering"],
            "vector_database": ["vector database", "pinecone", "weaviate", "qdrant", "milvus", "vector index", "nearest neighbor"]
        }
        
        # Target role title weights for relevance
        self.title_weights = {
            "senior ai engineer": 1.0,
            "founding senior ai engineer": 1.0,
            "senior ml engineer": 0.95,
            "lead ai engineer": 0.95,
            "staff machine learning engineer": 0.95,
            "recommendation systems engineer": 0.90,
            "search engineer": 0.90,
            "applied ml engineer": 0.85,
            "machine learning engineer": 0.80,
            "nlp engineer": 0.80,
            "senior data scientist": 0.75,
            "applied scientist": 0.80,
            "software engineer": 0.40,
            "backend engineer": 0.40,
            "data engineer": 0.35,
            "analytics engineer": 0.30,
            "operations manager": 0.0,
            "accountant": 0.0,
            "marketing manager": 0.0,
            "hr manager": 0.0,
            "customer support": 0.0
        }

    def extract_features(self, cand):
        """Extracts a dictionary of features from a single candidate profile."""
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        features = {}
        
        # ==========================================
        # 1. Experience Features
        # ==========================================
        yoe = profile.get("years_of_experience", 0)
        features["years_experience"] = yoe
        
        # Experience match score (bell curve around 5-9 years)
        if 5.0 <= yoe <= 9.0:
            features["experience_match_score"] = 1.0
        elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
            features["experience_match_score"] = 0.8
        elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 13.0:
            features["experience_match_score"] = 0.5
        else:
            features["experience_match_score"] = 0.2
            
        # Seniority match score
        current_title = profile.get("current_title", "").lower()
        is_senior_title = any(word in current_title for word in ["senior", "lead", "staff", "principal", "head", "manager"])
        features["seniority_match_score"] = 1.0 if is_senior_title else 0.5
        
        # ==========================================
        # 2. Skill Features
        # ==========================================
        # Calculate scores for each skill family, weighted by proficiency, duration, and endorsements
        for family, keywords in self.skill_keywords.items():
            family_score = 0.0
            for s in skills:
                name = s.get("name", "").lower()
                prof = s.get("proficiency", "beginner")
                dur = s.get("duration_months", 0)
                endorsements = s.get("endorsements", 0)
                
                # Maps proficiency to weight
                prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.8, "expert": 1.0}[prof]
                
                # Check if skill matches any keyword in the family
                max_kw_weight = 0.0
                for kw in keywords:
                    if kw in name:
                        max_kw_weight = 1.0
                        break
                        
                if max_kw_weight > 0.0:
                    # Duration weight (capped at 5 years / 60 months)
                    dur_weight = min(dur / 60.0, 1.0) if dur > 0 else 0.2
                    # Endorsement weight (log-scaled to prevent runaway scores)
                    import math
                    endorsement_weight = 1.0 + 0.1 * math.log1p(endorsements)
                    
                    skill_val = prof_weight * (0.5 + 0.5 * dur_weight) * endorsement_weight
                    family_score += skill_val
                    
            # Normalize family score
            features[f"{family}_skills"] = min(family_score / 3.0, 1.0)
            
        # ==========================================
        # 3. Career Features
        # ==========================================
        # Title relevance score based on target titles
        max_title_relevance = 0.0
        for kw, weight in self.title_weights.items():
            if kw in current_title:
                if weight > max_title_relevance:
                    max_title_relevance = weight
        features["title_relevance"] = max_title_relevance
        
        # Company quality and product vs service split
        has_product = False
        has_services = False
        services_companies = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini"]
        
        for job in career:
            comp = job.get("company", "").lower()
            ind = job.get("industry", "").lower()
            is_service_comp = any(sc in comp for sc in services_companies) or "services" in ind or "consulting" in ind
            if is_service_comp:
                has_services = True
            else:
                has_product = True
                
        if has_product and not has_services:
            features["product_vs_service_experience"] = 1.0  # Product-only
            features["company_quality"] = 1.0
        elif has_product and has_services:
            features["product_vs_service_experience"] = 0.7  # Product-service mix
            features["company_quality"] = 0.8
        else:
            features["product_vs_service_experience"] = 0.0  # Services-only
            features["company_quality"] = 0.2
            
        # Promotion progression (upward title trajectory)
        # Check if they went from Junior/Associate -> Senior/Lead/Staff over their career history
        progression = 0.5  # Neutral default
        if len(career) >= 2:
            # Career history is sorted reverse chronologically (newest first)
            titles = [job.get("title", "").lower() for job in career]
            # Check if older jobs (later in list) were junior and newer jobs (earlier in list) are senior
            older_is_junior = any(any(kw in title for kw in ["junior", "associate", "intern", "engineer"]) for title in titles[1:])
            newer_is_senior = any(any(kw in title for kw in ["senior", "lead", "staff", "principal", "head", "manager"]) for title in titles[:1])
            if older_is_junior and newer_is_senior:
                progression = 1.0  # Upward progression
            elif newer_is_senior:
                progression = 0.8  # Started senior or stable senior
                
        features["promotion_progression"] = progression
        
        # ==========================================
        # 4. Engagement Features
        # ==========================================
        features["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0.0)
        features["interview_completion_rate"] = signals.get("interview_completion_rate", 0.0)
        features["recruiter_saves"] = signals.get("saved_by_recruiters_30d", 0)
        features["search_appearances"] = signals.get("search_appearance_30d", 0)
        
        # ==========================================
        # 5. Availability Features
        # ==========================================
        features["notice_period"] = signals.get("notice_period_days", 180)
        features["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0
        features["relocation_willingness"] = 1.0 if signals.get("willing_to_relocate", False) else 0.0
        
        # ==========================================
        # 6. Quality Features
        # ==========================================
        features["github_activity"] = signals.get("github_activity_score", -1.0)
        features["profile_completeness"] = signals.get("profile_completeness_score", 100.0)
        
        # Career consistency (Crucial metric that separates genuine rare profiles from boilerplate fillers!)
        # Calculated as: 1.0 - fraction of jobs with boilerplate descriptions
        boilerplate_count = 0
        total_jobs = len(career)
        
        for job in career:
            desc = job.get("description", "")
            if desc in self.boilerplate_descriptions:
                boilerplate_count += 1
                
        features["career_consistency"] = 1.0 - (boilerplate_count / total_jobs) if total_jobs > 0 else 0.0
        
        return features
