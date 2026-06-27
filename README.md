# Redrob Hackathon: Intelligent Candidate Discovery System
## Founding Senior AI Engineer Ranker

This repository contains a production-grade, highly optimized, and explainable **Intelligent Candidate Discovery & Ranking System** designed to solve the Redrob Hackathon talent acquisition challenge. 

Our system streams a 100,000-candidate dataset, filters out 99.8% background noise and anomalous records (honeypots), scores candidates using a mathematically rigorous formula, and dynamically generates human-readable justifications for the top 100 candidates—all in **under 66 seconds** on a single CPU core with less than **500 MB RAM**.

---

## 1. Project Directory Structure

```
├── rank.py                       # Main CLI Entry Point
├── app.py                        # Streamlit Interactive Dashboard (Sandbox Demo)
├── validate_submission.py        # Official Challenge Validator
├── candidates.json               # Raw Candidate Pool (100,000 lines, JSONL)
├── job_description.docx          # Target Job Description (OpenXML)
├── submission.csv                # Output Ranked CSV File
├── submission_metadata.yaml      # Hackathon Participant Metadata
├── src/                          # Modular Source Code Package
│   ├── __init__.py
│   ├── jd_parser.py              # Phase 1: Job Description Intelligence Engine
│   ├── feature_engineering.py    # Phase 2: Candidate Feature Extractor
│   ├── retrieval.py              # Phase 3: Hybrid Retrieval Layer
│   ├── ranker.py                 # Phase 4: LTR-Inspired Weighted Ranker
│   ├── explainability.py         # Phase 5: Explainability & Reasoning Engine
│   ├── data_quality.py           # Phase 6: Anomaly & Honeypot Detector
│   ├── evaluation.py             # Phase 8: Information Retrieval Evaluator
│   └── utils.py                  # Stream loaders, date parsers, logging helpers
└── docs/                         # Technical Documentation & Deliverables
    ├── architecture.md           # System Architecture & Mermaid Diagrams
    ├── methodology_report.md     # Mathematical Model & Trap Neutralization Report
    └── presentation.md           # Pitch Deck Slide Content & Notes
```

---

## 2. Requirements & Setup

The **core ranking pipeline** (`rank.py` and the `src/` modules) is built entirely on the Python Standard Library to ensure maximum execution speed, extreme reliability, and zero dependency overhead. There are **no external library dependencies** (like pandas, numpy, or scikit-learn), enabling instant sandbox deployment and a tiny docker footprint that satisfies the Stage 3 docker reproduction container perfectly.

The **optional Streamlit Dashboard** (`app.py`) uses lightweight, industry-standard packages (`streamlit` and `pandas`) to provide a visual interface and candidate analyzer dashboard.

### Prerequisites & Dependencies
- **Core Pipeline**: Python 3.7+ (Standard Library only — zero dependencies).
- **Streamlit Dashboard (Optional)**: Requires `streamlit` and `pandas` packages.
- **Operating System**: Windows, macOS, or Linux.


---

## 3. How to Reproduce

To execute the candidate discovery and ranking system and generate the validated `submission.csv` file, run the following command from the project root:

```bash
python rank.py --candidates ./candidates.json --out ./submission.csv
```

### Script Execution Log
During execution, the script performs a fast pre-computation pass, initializes the pipeline modules, executes retrieval, ranks candidates, writes the CSV, and runs the official validator:

```
==================================================
Initializing Intelligent Candidate Discovery System
==================================================
Phase 1: Loading Job Description...
Reading job description from ./job_description.docx...
Parsed JD Rules. Seniority Target: Senior, Experience: 5-9 years.
Saved parsed JD schema to ./docs/parsed_jd.json
Starting pre-computation pass to build description frequency map...
Pre-computation complete in 10.64 seconds.
Processed 100000 candidates. Found 27 boilerplate career descriptions.
Phase 2-6: Initializing Pipeline Layers...
Phase 3: Running Hybrid Retrieval Layer over 100,000 candidates...
Retrieval complete. Selected 179 candidates into the high-precision recall pool.
Phase 4: Running LTR-Inspired Weighted Ranker...
Ranking complete. Scored and ranked 179 candidates.
Phase 5 & 10: Generating human-readable reasonings and compiling top 100...
Writing top 100 candidates to ./submission.csv...
Submission CSV compiled successfully.
Running official validator on the generated CSV file...
Validation PASSED! The generated CSV is 100% compliant with the challenge rules.
==================================================
System execution completed in 65.59 seconds.
==================================================
```

---

## 4. Architectural Summary & Trap Defense

Our system is designed as an active defense against synthetic talent traps:
1. **Boilerplate Description Filter**: Scans description frequencies to identify the 27 template descriptions used to generate the 99,821 filler candidates. By filtering out candidates with $0.0$ career consistency (who have only worked in boilerplate roles), we immediately isolate the **179 genuine ML/Search candidates** in the pool.
2. **Honeypot Blocker**: Evaluates every skill. If a candidate claims `"expert"` or `"advanced"` proficiency in a skill with `0` months of duration, the candidate is flagged as a honeypot and assigned a score of `0.0`.
3. **Temporal Inconsistency Checks**: Validates that career `duration_months` matches the difference between `start_date` and `end_date`. It also detects overlapping current employers and careers starting before education, zeroing out all 448 tampered records.
4. **Weighted Ranking & Multipliers**: Combines static qualifications (skill relevance, title match, experience fit, location, and notice period) with multiplicative platform engagement scores (responsiveness, interview attendance, activity recency), ensuring the most qualified, available, and responsive talent rises to the top of the list.
5. **Deterministic Tie-Breaking**: Breaks score ties alphabetically by `candidate_id` in ascending order, guaranteeing strict compliance with the ranking validator.

---

## 5. Interactive Streamlit Dashboard (Sandbox Demo)

To fulfill the mandatory **Sandbox/Demo link** requirement (Section 10.5 of `submission_spec.md`), we have built an interactive dashboard under `app.py`. It provides a clean, visual interface to parse JDs, upload candidates, view metrics, analyze profiles, and download the validated CSV.

### How to Run Locally

1. **Install the dependencies**:
   ```bash
   pip install streamlit pandas
   ```

2. **Launch the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the Platform**:
   - The application will open automatically in your browser at `http://localhost:8501`.
   - You can parse any Job Description text dynamically.
   - Run the pipeline on the preloaded 50-candidate sample or upload custom JSON/JSONL pools.
   - Explore individual candidate scores, matching skills, career history, and dynamic justifications.
   - Download the final validated `submission.csv` with a single click.

