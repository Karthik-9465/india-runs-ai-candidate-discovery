import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.resolve()))

from src.data_quality import DataQualityEngine
from src.feature_engineering import CandidateFeatureExtractor
from src.retrieval import HybridRetrievalLayer

def main():
    base_dir = Path(__file__).parent
    candidates_path = base_dir / "candidates.json"
    output_path = base_dir / "sample_candidates.json"

    print("Building boilerplate set...")
    descriptions = []
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            cand = json.loads(line)
            for job in cand.get("career_history", []):
                desc = job.get("description", "")
                if desc:
                    descriptions.append(desc)
    desc_counts = Counter(descriptions)
    boilerplate_set = {desc for desc, freq in desc_counts.items() if freq > 100}
    print(f"Boilerplate descriptions found: {len(boilerplate_set)}")

    dq_engine = DataQualityEngine(current_date=datetime(2026, 6, 24))
    extractor = CandidateFeatureExtractor(boilerplate_descriptions=boilerplate_set)
    retriever = HybridRetrievalLayer(data_quality_engine=dq_engine, feature_extractor=extractor)

    genuine_candidates = []
    honeypot_candidates = []
    contradiction_candidates = []
    noise_candidates = []

    print("Scanning candidates for sample...")
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            is_blacklisted, penalty_mult, reasons = dq_engine.analyze_candidate(cand)
            
            if is_blacklisted:
                if any("Honeypot" in r for r in reasons):
                    if len(honeypot_candidates) < 10:
                        honeypot_candidates.append(cand)
                else:
                    if len(contradiction_candidates) < 10:
                        contradiction_candidates.append(cand)
            else:
                features = extractor.extract_features(cand)
                if features.get("career_consistency", 0.0) > 0.0:
                    # Compute retrieval score
                    bm25_score = retriever.compute_bm25_score(cand)
                    skill_overlap = retriever.compute_skill_overlap(features)
                    semantic_sim = retriever.compute_semantic_simulation(features)
                    retrieval_score = 0.3 * bm25_score + 0.3 * skill_overlap + 0.4 * semantic_sim
                    genuine_candidates.append((retrieval_score, cand))
                else:
                    if len(noise_candidates) < 10:
                        noise_candidates.append(cand)
                        
            # Early stop if we have enough of each category
            if len(genuine_candidates) >= 150 and len(honeypot_candidates) >= 10 and len(contradiction_candidates) >= 10:
                break

    print(f"Found {len(genuine_candidates)} genuine, {len(honeypot_candidates)} honeypot, {len(contradiction_candidates)} contradiction.")

    # Sort genuine candidates by score and take the top 30
    genuine_candidates.sort(key=lambda x: -x[0])
    top_genuine = [x[1] for x in genuine_candidates[:30]]

    # Combine into exactly 50 candidates
    sample_pool = top_genuine + honeypot_candidates[:10] + contradiction_candidates[:10]

    # If we don't have enough, pad with noise or whatever we have
    while len(sample_pool) < 50 and noise_candidates:
        sample_pool.append(noise_candidates.pop(0))

    print(f"Writing {len(sample_pool)} candidates to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_pool, f, indent=2, ensure_ascii=False)
    print("Sample generation complete!")

if __name__ == "__main__":
    main()
