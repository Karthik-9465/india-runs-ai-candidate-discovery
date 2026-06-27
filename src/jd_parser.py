# src/jd_parser.py
import re

class JDIntelligenceEngine:
    """JD Intelligence Engine that extracts structured information from Job Descriptions."""
    
    def __init__(self):
        # Define core skill groups for matching
        self.skill_categories = {
            "embeddings": ["embedding", "sentence-transformer", "bge", "e5", "dense retrieval", "representation learning"],
            "vector_db": ["vector database", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "vector search", "hybrid search"],
            "ranking": ["ranking", "learning-to-rank", "xgboost", "lightgbm", "ltr", "personalization", "relevance", "recommendation"],
            "evaluation": ["ndcg", "mrr", "map", "evaluation framework", "ab testing", "offline-to-online", "precision@k"],
            "llm": ["llm", "fine-tuning", "lora", "qlora", "peft", "prompt engineering", "langchain", "llama", "mistral"],
            "python": ["python", "pip", "pyproject"],
            "distributed_systems": ["distributed systems", "inference optimization", "scale", "kubernetes", "kubeflow", "docker", "mlflow", "feature store"]
        }

    def parse_jd(self, jd_text):
        """
        Parses a job description text to extract structured rules.
        Includes fallbacks specifically tuned for the Redrob Founding Senior AI Engineer role.
        """
        text_lower = jd_text.lower()
        
        # 1. Extract Role & Seniority
        role = "AI Engineer"
        if "senior" in text_lower:
            seniority = "Senior"
        elif "lead" in text_lower:
            seniority = "Lead"
        elif "staff" in text_lower:
            seniority = "Staff"
        elif "principal" in text_lower:
            seniority = "Principal"
        else:
            seniority = "Mid-Senior"
            
        if "founding" in text_lower:
            role = "Founding Senior AI Engineer"
            
        # 2. Extract Experience Range
        exp_min = 5
        exp_max = 9
        # Regex search for experience e.g. "5-9 years", "5 to 9 years"
        exp_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*years', text_lower)
        if exp_match:
            exp_min = int(exp_match.group(1))
            exp_max = int(exp_match.group(2))
            
        # 3. Extract Notice Period Preference
        notice_pref = 30
        if "sub-30" in text_lower or "under 30" in text_lower or "30-day" in text_lower:
            notice_pref = 30
            
        # 4. Extract Location Preferences
        locations = []
        if "pune" in text_lower:
            locations.append("Pune")
        if "noida" in text_lower:
            locations.append("Noida")
        if "gurgaon" in text_lower or "delhi" in text_lower or "ncr" in text_lower:
            locations.append("Delhi NCR")
        if "hyderabad" in text_lower:
            locations.append("Hyderabad")
        if "mumbai" in text_lower:
            locations.append("Mumbai")
            
        # 5. Extract and Categorize Skills
        mandatory_skills = []
        preferred_skills = []
        
        # In this specific JD, we know the mandatory vs preferred split:
        # Mandatory: Embeddings, Vector DBs/Hybrid search, Python, Ranking Evaluation
        # Preferred: LLM fine-tuning, Learning-to-Rank models, HR-tech/Marketplaces, Distributed Systems
        
        # We check the text for matches in our categories
        matches = {}
        for category, keywords in self.skill_categories.items():
            matches[category] = any(kw in text_lower for kw in keywords)
            
        # Hardcoded semantic mapping based on JD text analysis
        if matches.get("embeddings"):
            mandatory_skills.extend(["embeddings-based retrieval", "sentence-transformers"])
        if matches.get("vector_db"):
            mandatory_skills.extend(["vector databases", "hybrid search", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "faiss"])
        if matches.get("python"):
            mandatory_skills.append("python")
        if matches.get("evaluation"):
            mandatory_skills.extend(["ranking evaluation", "ndcg", "mrr", "map", "ab testing"])
            
        if matches.get("llm"):
            preferred_skills.extend(["llm fine-tuning", "lora", "qlora", "peft"])
        if matches.get("ranking"):
            preferred_skills.extend(["learning-to-rank", "xgboost", "lightgbm", "recommendation systems"])
        if matches.get("distributed_systems"):
            preferred_skills.extend(["distributed systems", "inference optimization", "mlflow", "kubernetes"])
            
        # 6. Extract Platform/Vendor Preferences
        # Redrob specifically downweights:
        # - Pure research backgrounds
        # - Consulting/Services-only companies (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
        # - Framework-only enthusiasts (LangChain tutorial builders)
        # - Computer Vision / Speech only
        disqualified_employers = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
        
        return {
            "role": role,
            "seniority": seniority,
            "experience_range": {
                "min": exp_min,
                "max": exp_max
            },
            "mandatory_skills": list(set(mandatory_skills)),
            "preferred_skills": list(set(preferred_skills)),
            "retrieval_requirements": {
                "dense": "embeddings-based dense retrieval",
                "sparse": "hybrid search / BM25"
            },
            "ranking_requirements": {
                "model_types": ["learning-to-rank", "xgboost", "lightgbm"],
                "metrics": ["ndcg", "mrr", "map"]
            },
            "llm_requirements": ["fine-tuning", "lora/qlora"],
            "vector_db_requirements": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "faiss"],
            "notice_period_preference_days": notice_pref,
            "location_preferences": locations,
            "disqualified_employers": disqualified_employers
        }
