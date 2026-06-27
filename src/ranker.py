# src/ranker.py
from datetime import datetime
from src.utils import parse_date

class CandidateRanker:
    """Learning-to-Rank inspired ranker that scores candidates based on a multi-component formula."""
    
    def __init__(self, current_date=None):
        self.current_date = current_date or datetime(2026, 6, 24)
        
        # Scoring weights (sum to 1.0)
        self.w_exp = 0.15
        self.w_title = 0.20
        self.w_skill = 0.25
        self.w_company = 0.15
        self.w_loc = 0.15
        self.w_notice = 0.10

    def score_candidate(self, candidate_data):
        """
        Scores a single candidate using our multi-component LTR-inspired formula.
        S(c) = (sum_i w_i * s_i) * prod_j m_j * penalty_multiplier
        """
        cand = candidate_data["candidate"]
        features = candidate_data["features"]
        dq_multiplier = candidate_data["penalty_multiplier"]
        
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        
        # 1. Experience Score (s_exp)
        s_exp = features.get("experience_match_score", 0.2)
        
        # 2. Title Relevance Score (s_title)
        s_title = features.get("title_relevance", 0.0)
        
        # 3. Skill Fit Score (s_skill)
        # We average the specific target skill scores engineered in Phase 2
        skill_keys = [
            "embeddings_skills", "vector_database_skills", 
            "ranking_skills", "retrieval_skills", "rag_skills", "llm_skills"
        ]
        skill_vals = [features.get(k, 0.0) for k in skill_keys]
        # Weight Python and general ML slightly
        python_skill = 1.0 if any(s.get("name", "").lower() == "python" for s in cand.get("skills", [])) else 0.5
        s_skill = 0.8 * (sum(skill_vals) / len(skill_vals)) + 0.2 * python_skill
        s_skill = min(s_skill, 1.0)
        
        # 4. Company Quality Score (s_company)
        s_company = features.get("company_quality", 0.5)
        # Apply job-hopping penalty if career_consistency shows issues, or if flagged in reasons
        has_job_hopping = any("Job-hopping" in r for r in candidate_data["reasons"])
        if has_job_hopping:
            s_company *= 0.8
            
        # 5. Location Score (s_loc)
        loc = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        willing_relocate = features.get("relocation_willingness", 0.0) == 1.0
        
        if "noida" in loc or "pune" in loc:
            s_loc = 1.0
        elif "delhi" in loc or "gurgaon" in loc or "ncr" in loc or "ghaziabad" in loc or "faridabad" in loc:
            s_loc = 0.9
        elif "mumbai" in loc or "hyderabad" in loc or "bangalore" in loc or "bengaluru" in loc or "chennai" in loc or "kolkata" in loc:
            s_loc = 0.8 if willing_relocate else 0.5
        elif "india" in country:
            s_loc = 0.7 if willing_relocate else 0.4
        else:
            # Outside India
            s_loc = 0.3 if willing_relocate else 0.0
            
        # 6. Notice Period Score (s_notice)
        notice = features.get("notice_period", 180)
        if notice <= 30:
            s_notice = 1.0
        elif notice <= 60:
            s_notice = 0.8
        elif notice <= 90:
            s_notice = 0.5
        else:
            s_notice = 0.2
            
        # Compute Weighted Base Score (sum_i w_i * s_i)
        base_score = (
            self.w_exp * s_exp +
            self.w_title * s_title +
            self.w_skill * s_skill +
            self.w_company * s_company +
            self.w_loc * s_loc +
            self.w_notice * s_notice
        )
        
        # ==========================================
        # Multiplicative Multipliers (m_j)
        # ==========================================
        # A. Recruiter Responsiveness Multiplier
        resp_rate = features.get("recruiter_response_rate", 0.0)
        m_resp = 0.5 + 0.5 * resp_rate
        
        # B. Interview Attendance Multiplier
        completion_rate = features.get("interview_completion_rate", 0.0)
        m_completion = 0.5 + 0.5 * completion_rate
        
        # C. Activity Recency Multiplier
        last_active = parse_date(signals.get("last_active_date", ""))
        if last_active:
            days_inactive = (self.current_date - last_active).days
            if days_inactive <= 30:
                m_recency = 1.2
            elif days_inactive <= 90:
                m_recency = 1.0
            elif days_inactive <= 180:
                m_recency = 0.8
            else:
                m_recency = 0.5
        else:
            m_recency = 0.5
            
        # D. GitHub Activity Multiplier
        github = features.get("github_activity", -1.0)
        if github > 70:
            m_git = 1.15
        elif github > 40:
            m_git = 1.05
        elif github == -1.0:
            m_git = 0.95
        else:
            m_git = 1.0
            
        # E. Open to Work Multiplier
        open_to_work = features.get("open_to_work", 0.0)
        m_open = 1.2 if open_to_work == 1.0 else 0.9
        
        # F. Recruiter Saves Multiplier
        saves = features.get("recruiter_saves", 0)
        m_saves = 1.0 + min(saves / 10.0, 0.1)
        
        # Combine all multipliers
        m_engagement = m_resp * m_completion * m_recency * m_git * m_open * m_saves
        
        # Final Score
        final_score = base_score * m_engagement * dq_multiplier
        
        return {
            "candidate_id": cand.get("candidate_id"),
            "name": profile.get("anonymized_name"),
            "title": profile.get("current_title"),
            "score": round(final_score, 4),
            "base_score": round(base_score, 4),
            "m_engagement": round(m_engagement, 4),
            "reasons": candidate_data["reasons"],
            "features": features,
            "candidate": cand
        }

    def rank_candidates(self, recall_pool):
        """Scores all candidates in the recall pool and sorts them by score descending."""
        scored_candidates = []
        for cand_data in recall_pool:
            scored = self.score_candidate(cand_data)
            # Filter out zero-score candidates (blacklisted/disqualified)
            if scored["score"] > 0.0:
                scored_candidates.append(scored)
                
        # Sort by:
        # 1. Final score descending
        # 2. Candidate ID ascending (to break ties deterministically as required by the validator!)
        scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        
        # Assign ranks 1 to N
        for i, cand in enumerate(scored_candidates):
            cand["rank"] = i + 1
            
        return scored_candidates
