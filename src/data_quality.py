# src/data_quality.py
from datetime import datetime
from src.utils import calculate_months_between, parse_date

class DataQualityEngine:
    """Data Quality Engine that detects anomalies, honeypots, and low-quality profiles."""
    
    def __init__(self, current_date=None):
        # Current local time of the hackathon evaluation (June 24, 2026)
        self.current_date = current_date or datetime(2026, 6, 24)
        
        # Services/consulting firms that are explicitly downweighted/disqualified if exclusive
        self.services_companies = [
            "tcs", "tata consultancy services", "infosys", "wipro", "cognizant", 
            "accenture", "capgemini", "hcl", "l&t", "mindtree", "mphasis", "tech mahindra"
        ]

    def analyze_candidate(self, cand):
        """
        Analyzes a candidate profile and returns:
        - is_blacklisted (bool): True if it's a honeypot or completely invalid.
        - penalty_multiplier (float): A score multiplier from 0.0 to 1.0.
        - reasons (list of str): Detailed explanations of detected issues.
        """
        cid = cand.get("candidate_id", "UNKNOWN")
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        reasons = []
        is_blacklisted = False
        penalty_multiplier = 1.0
        
        # 1. Check for Missing Core Fields
        if not profile:
            is_blacklisted = True
            reasons.append("Missing profile object")
            return True, 0.0, reasons
            
        if not career:
            is_blacklisted = True
            reasons.append("Missing career history")
            return True, 0.0, reasons

        # 2. Detect Honeypots (Expert/Advanced skills with 0 duration)
        expert_zero = [s["name"] for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
        if expert_zero:
            is_blacklisted = True
            reasons.append(f"Honeypot: Expert skill(s) with 0 duration: {', '.join(expert_zero)}")
            
        # In synthetic datasets, sometimes advanced skills with 0 duration are also honeypot traps
        advanced_zero = [s["name"] for s in skills if s.get("proficiency") == "advanced" and s.get("duration_months", 0) == 0]
        if advanced_zero:
            is_blacklisted = True
            reasons.append(f"Honeypot: Advanced skill(s) with 0 duration: {', '.join(advanced_zero)}")

        # 3. Career History Date Inconsistencies (Contradictory Experience)
        date_mismatch = False
        mismatched_jobs = []
        for job in career:
            start = job.get("start_date")
            end = job.get("end_date")
            dur = job.get("duration_months", 0)
            if start:
                expected_dur = calculate_months_between(start, end, self.current_date)
                # Allow a small buffer of 3 months for date rounding
                if abs(expected_dur - dur) > 3:
                    date_mismatch = True
                    mismatched_jobs.append(f"{job.get('company')} ({job.get('title')}): listed {dur}m, actual {expected_dur}m")
                    
        if date_mismatch:
            is_blacklisted = True
            reasons.append(f"Logical contradiction: Career duration mismatch: {'; '.join(mismatched_jobs)}")

        # 4. Experience & YOE Contradictions
        yoe = profile.get("years_of_experience", 0)
        total_history_months = sum(job.get("duration_months", 0) for job in career)
        total_history_years = total_history_months / 12.0
        
        # If total career history length is way larger than stated YOE (e.g. history is 15 years, but YOE is 2)
        if total_history_years > yoe + 5 and yoe > 0:
            is_blacklisted = True
            reasons.append(f"Logical contradiction: Career history length ({total_history_years:.1f} yrs) significantly exceeds YOE ({yoe} yrs)")

        # 5. Overlapping Current Jobs
        current_jobs = [job for job in career if job.get("is_current")]
        if len(current_jobs) > 1:
            is_blacklisted = True
            reasons.append(f"Logical contradiction: Multiple simultaneous current jobs: {', '.join(j.get('company') for j in current_jobs)}")

        # 6. Career starting before Education (Unrealistic career timeline)
        education = cand.get("education", [])
        if education and career:
            earliest_job_year = 9999
            for job in career:
                start = job.get("start_date")
                if start:
                    dt = parse_date(start)
                    if dt and dt.year < earliest_job_year:
                        earliest_job_year = dt.year
            
            earliest_edu_year = 9999
            for edu in education:
                start_yr = edu.get("start_year")
                if start_yr and start_yr < earliest_edu_year:
                    earliest_edu_year = start_yr
                    
            if earliest_job_year < earliest_edu_year - 5 and earliest_edu_year != 9999:
                is_blacklisted = True
                reasons.append(f"Logical contradiction: Career started in {earliest_job_year}, before education in {earliest_edu_year}")

        # 7. Services-Only Career (Low quality for startup founding team)
        # Check if the candidate has ONLY worked in IT Services / Consulting
        has_product = False
        has_services = False
        
        for job in career:
            comp = job.get("company", "").lower()
            ind = job.get("industry", "").lower()
            
            is_service_comp = any(sc in comp for sc in self.services_companies) or "services" in ind or "consulting" in ind
            if is_service_comp:
                has_services = True
            else:
                has_product = True
                
        if has_services and not has_product:
            # Services-only penalty (Not blacklisted, but heavily penalized)
            penalty_multiplier *= 0.2
            reasons.append("Services-only background: Candidate has only worked at IT services/consulting firms")
            
        # 8. Job-Hopping (Title-Chaser Penalty)
        # Optimizing for Senior -> Staff -> Principal by switching companies every 1.5 years
        if len(career) >= 3:
            total_months = sum(job.get("duration_months", 0) for job in career)
            avg_tenure_months = total_months / len(career)
            # If average tenure is less than 18 months, apply job-hopping penalty
            if avg_tenure_months < 18.0:
                penalty_multiplier *= 0.8
                reasons.append(f"Job-hopping behavior: Average job tenure of {avg_tenure_months:.1f} months is under 18 months")

        # 9. Low Completeness Score
        completeness = signals.get("profile_completeness_score", 100.0)
        if completeness < 40.0:
            penalty_multiplier *= 0.7
            reasons.append(f"Low profile completeness: {completeness}%")

        if is_blacklisted:
            penalty_multiplier = 0.0

        return is_blacklisted, penalty_multiplier, reasons
