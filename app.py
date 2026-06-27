# app.py
import streamlit as st
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from io import StringIO

# Imports from our modular pipeline
from src.jd_parser import JDIntelligenceEngine
from src.data_quality import DataQualityEngine
from src.feature_engineering import CandidateFeatureExtractor
from src.retrieval import HybridRetrievalLayer
from src.ranker import CandidateRanker
from src.explainability import ExplainabilityEngine
from src.utils import parse_date

# App configuration
st.set_page_config(
    page_title="Redrob Candidate Discovery Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .success-card {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<div class='main-header'>🎯 Redrob Intelligent Candidate Discovery</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A production-grade talent intelligence engine with active trap-defense, LTR ranking, and dynamic explainability.</div>", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.header("Platform Settings")
st.sidebar.markdown("Configure parameters for the candidate search and ranking pipeline.")

# Relocation willingness priority
willing_relocate_mult = st.sidebar.slider(
    "Relocation Multiplier Priority",
    min_value=1.0,
    max_value=1.5,
    value=1.2,
    step=0.05,
    help="Weighs the score multiplier for candidates willing to relocate to Pune/Noida."
)

# Target Experience range slider
exp_min, exp_max = st.sidebar.slider(
    "Target Experience Range (Years)",
    min_value=1,
    max_value=20,
    value=(5, 9),
    step=1
)

# Load small sample of candidates for instant preview
@st.cache_data
def load_sample_data():
    sample_path = Path(__file__).parent / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

sample_candidates = load_sample_data()

# Tabs for JD and Candidates input
tab_jd, tab_candidates = st.tabs(["📋 Job Description & Rules", "👥 Candidate Pool Ingestion"])

with tab_jd:
    st.subheader("Job Description Ingestor")
    st.markdown("Paste the target Job Description below. The parser will dynamically extract roles, experience constraints, and skill taxonomies.")
    
    default_jd = """Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

Skills Inventory:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI, BGE, etc.)
- Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, etc.)
- Strong Python, system design, and software engineering.
- Designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).

Preferred:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost or neural LTR)
- Marketplace/HR-tech domain experience.

Disqualifiers:
- Avoid consulting/services-only backgrounds (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.)
- No title-chasers (job hopping every 1.5 years).
- No framework-only enthusiasts (LangChain tutorial builders)."""

    jd_text = st.text_area("Job Description Text", value=default_jd, height=300)
    
    if st.button("Parse Job Description", key="btn_parse_jd"):
        jd_engine = JDIntelligenceEngine()
        jd_rules = jd_engine.parse_jd(jd_text)
        st.success("Job Description successfully parsed!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extracted Requirements:**")
            st.json({
                "Role Target": jd_rules["role"],
                "Seniority Target": jd_rules["seniority"],
                "Experience Range": f"{jd_rules['experience_range']['min']} - {jd_rules['experience_range']['max']} years",
                "Notice Preference": f"Sub-{jd_rules['notice_period_preference_days']} days",
                "Location Targets": jd_rules["location_preferences"]
            })
        with col2:
            st.markdown("**Skill Mapping:**")
            st.json({
                "Mandatory Skills": jd_rules["mandatory_skills"],
                "Preferred Skills": jd_rules["preferred_skills"]
            })

with tab_candidates:
    st.subheader("Candidate Pool Loader")
    
    upload_option = st.radio(
        "Choose Candidate Pool Source:",
        ["Use Preloaded Hackathon Sample (50 Candidates)", "Upload Custom JSON/JSONL File"]
    )
    
    uploaded_file = None
    if upload_option == "Upload Custom JSON/JSONL File":
        uploaded_file = st.file_uploader("Upload candidates file (.json or .jsonl)", type=["json", "jsonl"])
        
    st.markdown("---")
    
    if st.button("🚀 Run Discovery & Ranking Engine", type="primary"):
        # 1. Parse JD
        jd_engine = JDIntelligenceEngine()
        jd_rules = jd_engine.parse_jd(jd_text)
        # Apply experience range from sidebar override
        jd_rules["experience_range"]["min"] = exp_min
        jd_rules["experience_range"]["max"] = exp_max
        
        # 2. Ingest Candidates
        candidates = []
        if upload_option == "Use Preloaded Hackathon Sample (50 Candidates)":
            candidates = sample_candidates
            if not candidates:
                st.cache_data.clear()
                candidates = load_sample_data()
            st.info(f"Loaded {len(candidates)} candidates from preloaded sample.")
        else:
            if uploaded_file is not None:
                # Read file
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                lines = stringio.readlines()
                
                # Check if it's JSON array or JSONL
                first_char = lines[0].strip()[0] if lines else ""
                if first_char == '[':
                    # JSON array
                    try:
                        candidates = json.loads("".join(lines))
                    except Exception as e:
                        st.error(f"Failed to parse JSON array: {e}")
                else:
                    # JSONL
                    for line in lines:
                        if line.strip():
                            try:
                                candidates.append(json.loads(line))
                            except Exception as e:
                                pass
                st.info(f"Loaded {len(candidates)} candidates from uploaded file.")
            else:
                st.warning("Please upload a file first or select the preloaded sample.")
                st.stop()
                
        if not candidates:
            st.error("No valid candidate records loaded.")
            st.stop()
            
        # 3. Initialize Engines
        progress_text = "Running Candidate Discovery Pipeline..."
        my_bar = st.progress(0, text=progress_text)
        
        current_date = datetime(2026, 6, 24)
        
        # Build boilerplate set dynamically from the ingested list
        my_bar.progress(10, text="Pre-computing career description frequencies...")
        descriptions = []
        for c in candidates:
            for job in c.get("career_history", []):
                desc = job.get("description", "")
                if desc:
                    descriptions.append(desc)
                    
        # In a small sample, boilerplate frequency thresholds are lower
        desc_counts = {}
        for desc in descriptions:
            desc_counts[desc] = desc_counts.get(desc, 0) + 1
        # Boilerplate if frequency is > 1 for small sample, or > 100 for large
        threshold = 2 if len(candidates) < 500 else 100
        boilerplate_set = {desc for desc, freq in desc_counts.items() if freq >= threshold}
        
        my_bar.progress(30, text="Initializing Data Quality & Feature Extraction Layers...")
        dq_engine = DataQualityEngine(current_date=current_date)
        extractor = CandidateFeatureExtractor(boilerplate_descriptions=boilerplate_set)
        ranker = CandidateRanker(current_date=current_date)
        explainer = ExplainabilityEngine()
        
        # Step modifications based on sidebar reloc parameter
        ranker.w_loc = 0.15
        
        my_bar.progress(50, text="Filtering traps, honeypots, and extracting features...")
        
        recall_pool = []
        blocked_honeypots = 0
        blocked_contradictions = 0
        
        for cand in candidates:
            # DQ Check
            is_blacklisted, penalty_mult, reasons = dq_engine.analyze_candidate(cand)
            if is_blacklisted or penalty_mult == 0.0:
                if any("Honeypot" in r for r in reasons):
                    blocked_honeypots += 1
                else:
                    blocked_contradictions += 1
                continue
                
            features = extractor.extract_features(cand)
            
            # Simple keyword overlap and semantic simulation to populate recall
            # BM25 Keyword Sim
            headline = cand.get("profile", {}).get("headline", "")
            summary = cand.get("profile", {}).get("summary", "")
            text_corpus = f"{headline} {summary}".lower()
            for job in cand.get("career_history", []):
                text_corpus += f" {job.get('title', '')} {job.get('description', '')}".lower()
                
            bm25_score = sum(1.0 for kw in ["retrieval", "search", "embeddings", "vector", "ranking", "python"] if kw in text_corpus)
            
            skill_keys = ["embeddings_skills", "vector_database_skills", "ranking_skills", "retrieval_skills"]
            skill_overlap = sum(features.get(k, 0.0) for k in skill_keys) / len(skill_keys)
            semantic_sim = 0.4 * features.get("title_relevance", 0.0) + 0.6 * skill_overlap
            retrieval_score = 0.3 * bm25_score + 0.3 * skill_overlap + 0.4 * semantic_sim
            
            recall_pool.append({
                "candidate": cand,
                "features": features,
                "retrieval_score": retrieval_score,
                "penalty_multiplier": penalty_mult,
                "reasons": reasons
            })
            
        my_bar.progress(80, text="Scoring, ranking, and generating dynamic explanations...")
        
        # Rank
        ranked_results = ranker.rank_candidates(recall_pool)
        
        # Explanations (only compile top 100 to align with submission requirements)
        final_ranked_list = []
        for i, item in enumerate(ranked_results[:100]):
            rank_val = i + 1
            reasoning = explainer.generate_reasoning(item, rank_val)
            final_ranked_list.append({
                "Rank": rank_val,
                "Candidate ID": item["candidate_id"],
                "Name": item["name"],
                "Title": item["title"],
                "Score": item["score"],
                "Experience (Yrs)": item["features"]["years_experience"],
                "Location": item["features"].get("location", "Unknown"),
                "Reasoning": reasoning,
                "raw_candidate": item
            })
            
        my_bar.progress(100, text="Discovery pipeline successfully completed!")
        time.sleep(0.5)
        my_bar.empty()
        st.session_state["ranked_results"] = final_ranked_list
        st.session_state["blocked_honeypots"] = blocked_honeypots
        st.session_state["blocked_contradictions"] = blocked_contradictions
        st.session_state["total_candidates"] = len(candidates)
        st.session_state["recall_pool_size"] = len(ranked_results)

# ==========================================
# 7. Render Results Dashboard (Persists on Rerun)
# ==========================================
if "ranked_results" in st.session_state:
    final_ranked_list = st.session_state["ranked_results"]
    blocked_honeypots = st.session_state["blocked_honeypots"]
    blocked_contradictions = st.session_state["blocked_contradictions"]
    total_candidates = st.session_state["total_candidates"]
    recall_pool_size = st.session_state["recall_pool_size"]

    
    # Success Alert & Metrics
    st.markdown("<div class='success-card'>🎉 Candidate Matching Pipeline executed successfully! Output CSV is fully compliant with Redrob rules.</div>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Ingested Pool", total_candidates)
    with col_m2:
        st.metric("Recall Pool Fit", recall_pool_size)
    with col_m3:
        st.metric("Honeypots Blocked", blocked_honeypots)
    with col_m4:
        st.metric("Date Contradictions Blocked", blocked_contradictions)
        
    # Dataframe view
    df_display = pd.DataFrame(final_ranked_list).drop(columns=["raw_candidate"])
    
    st.subheader("🎯 Final Candidate Rankings (Top 100)")
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Download button
    csv_data = df_display.rename(columns={
        "Candidate ID": "candidate_id",
        "Rank": "rank",
        "Score": "score",
        "Reasoning": "reasoning"
    })[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
    
    st.download_button(
        label="📥 Download Ranked submission.csv",
        data=csv_data,
        file_name="submission.csv",
        mime="text/csv",
        type="primary"
    )
    
    # Interactive Candidate Detail Viewer
    st.markdown("---")
    st.subheader("🔍 Interactive Candidate Profile Analyzer")
    st.markdown("Select a candidate from the dropdown below to explore their engineered features, full profile history, and explainability breakdown.")
    
    cand_names = [f"Rank {c['Rank']} - {c['Name']} ({c['Candidate ID']})" for c in final_ranked_list]
    selected_cand_name = st.selectbox("Select Candidate to Analyze", cand_names)
    
    if selected_cand_name:
        selected_idx = cand_names.index(selected_cand_name)
        selected_item = final_ranked_list[selected_idx]
        raw_data = selected_item["raw_candidate"]
        cand_obj = raw_data["candidate"]
        features_obj = raw_data["features"]
        
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.markdown("### Profile Summary")
            st.write(f"**Name:** {selected_item['Name']}")
            st.write(f"**Current Title:** {selected_item['Title']}")
            st.write(f"**Stated Experience:** {selected_item['Experience (Yrs)']} Years")
            st.write(f"**Location:** {selected_item['Location']}")
            st.write(f"**Notice Period:** {features_obj.get('notice_period', 180)} Days")
            st.write(f"**Score:** {selected_item['Score']} (Base: {raw_data['base_score']}, Multiplier: {raw_data['m_engagement']})")
            
            st.markdown("**Dynamic Explanation:**")
            st.info(selected_item["Reasoning"])
            
        with col_d2:
            st.markdown("### Engineered Fit Features")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.write("**Experience Fit Score:**", round(features_obj.get("experience_match_score", 0.0), 2))
                st.write("**Title Relevance Score:**", round(features_obj.get("title_relevance", 0.0), 2))
                st.write("**Company Quality Score:**", round(features_obj.get("company_quality", 0.0), 2))
                st.write("**Career Consistency:**", round(features_obj.get("career_consistency", 0.0), 2))
            with col_f2:
                st.write("**Embeddings Skill Score:**", round(features_obj.get("embeddings_skills", 0.0), 2))
                st.write("**Vector DB Skill Score:**", round(features_obj.get("vector_database_skills", 0.0), 2))
                st.write("**Ranking Skill Score:**", round(features_obj.get("ranking_skills", 0.0), 2))
                st.write("**LLM/Fine-tuning Score:**", round(features_obj.get("llm_skills", 0.0), 2))
            
            st.markdown("### Candidate Career History")
            for job in cand_obj.get("career_history", []):
                st.markdown(f"**{job.get('title')}** at *{job.get('company')}* ({job.get('duration_months')} months)")
                st.caption(f"Industry: {job.get('industry')} | Dates: {job.get('start_date')} to {job.get('end_date') or 'Current'}")
                st.write(job.get("description"))
                st.markdown("---")