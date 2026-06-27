# src/retrieval.py
import re

class HybridRetrievalLayer:
    """Hybrid Retrieval Layer implementing Rule-based, BM25, Semantic, and Skill retrieval."""
    
    def __init__(self, data_quality_engine, feature_extractor):
        self.dq_engine = data_quality_engine
        self.extractor = feature_extractor
        
        # Target keywords for BM25-style scoring
        self.target_keywords = {
            "retrieval": 2.5,
            "search": 2.0,
            "embeddings": 2.5,
            "vector": 2.0,
            "ranking": 2.5,
            "recommendation": 2.0,
            "learning-to-rank": 2.5,
            "ltr": 2.0,
            "rag": 2.0,
            "fine-tuning": 1.5,
            "lora": 1.5,
            "qlora": 1.5,
            "peft": 1.5,
            "xgboost": 1.5,
            "lightgbm": 1.5,
            "ndcg": 2.0,
            "mrr": 2.0,
            "map": 2.0,
            "ab testing": 1.5,
            "python": 1.0
        }

    def compute_bm25_score(self, cand):
        """Simulates a BM25 keyword matching score over candidate text fields."""
        profile = cand.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        # Combine text fields to search
        text_corpus = f"{headline} {summary}"
        for job in cand.get("career_history", []):
            text_corpus += f" {job.get('title', '')} {job.get('description', '')}"
            
        text_lower = text_corpus.lower()
        score = 0.0
        
        for kw, weight in self.target_keywords.items():
            # Count term frequency (tf)
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
            if count > 0:
                # Simulates BM25 saturation: tf / (tf + k1 * (1 - b + b * (L / avgL)))
                # Simplified: tf / (tf + 1.2)
                tf_sat = count / (count + 1.2)
                score += tf_sat * weight
                
        return score

    def compute_skill_overlap(self, features):
        """Computes skill overlap score based on engineered skill features."""
        # Sum of specific target skill features engineered in Phase 2
        skills = [
            features.get("embeddings_skills", 0.0),
            features.get("vector_database_skills", 0.0),
            features.get("ranking_skills", 0.0),
            features.get("retrieval_skills", 0.0),
            features.get("rag_skills", 0.0)
        ]
        if not skills:
            return 0.0
        return sum(skills) / len(skills)

    def compute_semantic_simulation(self, features):
        """Simulates semantic embedding similarity score using domain relevance features."""
        # Combines experience, title relevance, and technical skill scores
        title_rel = features.get("title_relevance", 0.0)
        skill_score = (
            features.get("embeddings_skills", 0.0) * 0.25 +
            features.get("vector_database_skills", 0.0) * 0.25 +
            features.get("ranking_skills", 0.0) * 0.25 +
            features.get("retrieval_skills", 0.0) * 0.25
        )
        # Cosine-like combination
        return 0.4 * title_rel + 0.6 * skill_score

    def retrieve_recall_pool(self, candidates_generator, max_pool_size=500):
        """
        Streams candidates and retrieves a high-precision recall pool.
        Filters out blacklisted and boilerplate filler background candidates.
        """
        recall_pool = []
        
        for cand in candidates_generator:
            cid = cand.get("candidate_id")
            
            # A. Rule-based Retrieval (Stage 1: Anomaly & Quality check)
            is_blacklisted, penalty_mult, reasons = self.dq_engine.analyze_candidate(cand)
            if is_blacklisted or penalty_mult == 0.0:
                continue
                
            # Extract features (Phase 2)
            features = self.extractor.extract_features(cand)
            
            # Rule-based filter: Disqualify background filler profiles
            # A genuine candidate MUST have a career consistency > 0.0
            # (i.e. they must have at least one custom, non-boilerplate description)
            if features.get("career_consistency", 0.0) == 0.0:
                continue
                
            # B. BM25 keyword score
            bm25_score = self.compute_bm25_score(cand)
            
            # C. Skill overlap score
            skill_overlap = self.compute_skill_overlap(features)
            
            # D. Semantic simulation score
            semantic_sim = self.compute_semantic_simulation(features)
            
            # Combine into a retrieval score to rank the recall pool
            retrieval_score = 0.3 * bm25_score + 0.3 * skill_overlap + 0.4 * semantic_sim
            
            recall_pool.append({
                "candidate": cand,
                "features": features,
                "retrieval_score": retrieval_score,
                "penalty_multiplier": penalty_mult,
                "reasons": reasons
            })
            
        # Sort by retrieval score descending
        recall_pool.sort(key=lambda x: -x["retrieval_score"])
        
        # Limit to max_pool_size to ensure fast downstream ranking
        return recall_pool[:max_pool_size]
