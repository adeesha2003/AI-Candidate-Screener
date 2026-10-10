import os
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-fitted Domain TF-IDF model if available
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "domain_tfidf_model.pkl")
domain_tfidf_model = None
if os.path.exists(TFIDF_MODEL_PATH):
    try:
        domain_tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    except Exception as err:
        print(f"Warning: Could not load domain TF-IDF model: {err}")

# Lazily loaded SentenceTransformer model
_sbert_cache = None

def get_sbert_model():
    """Load fine-tuned domain SBERT model or fall back to base encoder."""
    global _sbert_cache
    if _sbert_cache is None:
        fine_tuned_path = os.path.join(os.path.dirname(__file__), "final-hr-model")
        if os.path.exists(fine_tuned_path):
            try:
                _sbert_cache = SentenceTransformer(fine_tuned_path)
            except Exception:
                _sbert_cache = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            _sbert_cache = SentenceTransformer('all-MiniLM-L6-v2')
    return _sbert_cache

# Domain-specific non-informative words
HR_STOPWORDS = {
    'passionate', 'seeking', 'opportunity', 'team', 'player', 
    'responsibilities', 'duties', 'highly', 'motivated', 'driven', 
    'excellent', 'skills'
}

# Domain competency clusters (143+ industry skills across 4 engineering clusters)
SKILL_CLUSTERS = {
    'ai_data': {
        'python', 'machine learning', 'data science', 'deep learning', 'pytorch', 
        'tensorflow', 'nlp', 'natural language processing', 'scikit-learn', 'pandas', 
        'sql', 'tableau', 'data analysis', 'fastapi', 'numpy', 'scipy', 'keras', 
        'opencv', 'matplotlib', 'seaborn', 'nltk', 'spacy', 'hugging face', 'huggingface', 
        'transformers', 'llm', 'bert', 'xgboost', 'lightgbm', 'pyspark', 'spark', 
        'hadoop', 'hive', 'power bi', 'bigquery', 'snowflake', 'statistics', 'r', 
        'data modeling', 'etl', 'data engineering', 'data warehouse', 'airflow'
    },
    'frontend': {
        'javascript', 'typescript', 'react', 'html', 'html5', 'css', 'css3', 'redux', 
        'vue', 'vue.js', 'angular', 'frontend', 'web development', 'next.js', 'nextjs', 
        'nuxt', 'tailwind', 'bootstrap', 'sass', 'scss', 'webpack', 'vite', 'jquery', 
        'graphql', 'responsive', 'dom', 'es6', 'jest', 'cypress', 'ui', 'ux'
    },
    'backend': {
        'java', 'spring', 'spring boot', '.net', 'c#', 'c++', 'microservices', 
        'sql', 'api', 'rest', 'restful', 'flask', 'fastapi', 'backend', 'django', 
        'node.js', 'nodejs', 'express', 'ruby', 'rails', 'php', 'laravel', 'golang', 
        'go', 'rust', 'asp.net', 'hibernate', 'jpa', 'soap', 'grpc', 'kafka', 
        'rabbitmq', 'celery', 'redis', 'postgresql', 'postgres', 'mysql', 'mongodb', 
        'cassandra', 'dynamodb', 'sqlite', 'oracle'
    },
    'devops_cloud': {
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'ci/cd', 'jenkins', 
        'terraform', 'linux', 'git', 'sysadmin', 'cloud', 'helm', 'ansible', 
        'gitlab', 'github actions', 'prometheus', 'grafana', 'elk', 'splunk', 
        'nginx', 'apache', 'bash', 'shell', 'powershell', 'ubuntu', 'centos', 'rhel'
    }
}

CORE_SKILLS = set().union(*SKILL_CLUSTERS.values())

# Career transferability rules (candidate pattern, target job pattern, credit boost)
CAREER_TRANSFERS = [
    (re.compile(r'\b(data\s*analyst|data\s*analytics|bi\s*analyst)\b', re.I),
     re.compile(r'\b(ai|machine\s*learning|data\s*science)\b', re.I), 20.0),
    (re.compile(r'\b(backend|java|spring|\.net)\b', re.I),
     re.compile(r'\b(frontend|react|web)\b', re.I), 7.5),
    (re.compile(r'\b(sysadmin|system\s*administrator|linux\s*admin)\b', re.I),
     re.compile(r'\b(devops|cloud)\b', re.I), 4.0),
]

SYNONYM_MAP = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence'
}

def parse_work_tenure(text):
    """Extract experience tenure using explicit phrases and calendar date spans."""
    lowered = text.lower()
    
    # Explicit mentions like '4 years', '3 yrs'
    explicit = re.findall(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)', lowered)
    explicit_tenure = max([int(x) for x in explicit if 1 <= int(x) <= 35], default=0)
    
    # Date spans (e.g. 2018 - 2022, 2020 - Present)
    date_regex = re.compile(
        r'(\b(?:19[89]\d|20[012]\d)\b)\s*(?:-|to|–|—)\s*(\b(?:19[89]\d|20[012]\d)\b|present|current)', 
        re.I
    )
    date_matches = date_regex.findall(text)
    
    span_tenure = 0
    if date_matches:
        current_year = 2026
        spans = []
        for start_s, end_s in date_matches:
            start_yr = int(start_s)
            end_yr = current_year if end_s.lower() in ['present', 'current'] else int(end_s)
            if 1990 <= start_yr <= current_year and start_yr <= end_yr <= current_year:
                spans.append((start_yr, end_yr))
        if spans:
            span_tenure = min(30, max(e for _, e in spans) - min(s for s, _ in spans))
            
    return max(explicit_tenure, span_tenure)

def clean_text_for_vectorizer(text):
    """Normalize text and boost domain terminology for sparse representation."""
    lowered = text.lower()
    for short_form, full_form in SYNONYM_MAP.items():
        lowered = re.sub(r'\b' + short_form + r'\b', full_form, lowered)
        
    for skill in CORE_SKILLS:
        if skill in lowered:
            lowered += f" {skill} {skill} {skill} {skill} {skill}"
            
    tokens = re.findall(r'\b\w+\b', lowered)
    clean_tokens = [t for t in tokens if t not in HR_STOPWORDS]
    return ' '.join(clean_tokens)

def compute_token_overlap(str_a, str_b):
    """Compute token Jaccard similarity (intersection over union)."""
    set_a = set(str_a.split())
    set_b = set(str_b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def split_into_windows(text, window_size=140, step_size=105):
    """Sliding-window chunker to bypass 256-token transformer truncation."""
    words = text.split()
    if len(words) <= window_size:
        return [text]
    chunks = []
    for i in range(0, len(words), step_size):
        chunks.append(" ".join(words[i:i + window_size]))
        if i + window_size >= len(words):
            break
    return chunks

def calculate_match_scores(job_description, resumes):
    """Execute hybrid screening ensemble and map calibrated scores to suitability tiers."""
    if not job_description or not resumes:
        return []

    required_experience = parse_work_tenure(job_description)
    
    # 1. TF-IDF Lexical Match
    cleaned_job = clean_text_for_vectorizer(job_description)
    cleaned_resumes = [clean_text_for_vectorizer(r) for r in resumes]
    
    if domain_tfidf_model is not None:
        job_vec = domain_tfidf_model.transform([cleaned_job])
        resume_vecs = domain_tfidf_model.transform(cleaned_resumes)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_data = vectorizer.fit_transform([cleaned_job] + cleaned_resumes)
        job_vec = tfidf_data[0:1]
        resume_vecs = tfidf_data[1:]

    cosine_sim_tfidf = cosine_similarity(job_vec, resume_vecs).flatten()

    # 2. Dense Semantic Match with Sliding-Window Chunking
    transformer = get_sbert_model()
    job_embedding = transformer.encode([job_description])
    
    sbert_scores = []
    for r in resumes:
        windows = split_into_windows(r)
        if len(windows) == 1:
            res_emb = transformer.encode([r])
            sbert_scores.append(float(cosine_similarity(job_embedding, res_emb)[0][0]))
        else:
            win_embeddings = transformer.encode(windows)
            win_sims = cosine_similarity(job_embedding, win_embeddings)[0]
            top2 = sorted(win_sims, reverse=True)[:2]
            sim_max_pool = float(0.70 * top2[0] + 0.30 * np.mean(top2))
            sbert_scores.append(sim_max_pool)

    # 3. Domain Skills Extraction
    job_tech_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', job_description.lower())}

    evaluated_candidates = []
    for idx, resume_text in enumerate(resumes):
        cand_tech_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', resume_text.lower())}
        
        # Bounded Skill Cluster Taxonomy Affinity
        skill_affinity = 0.0
        if job_tech_skills:
            exact_matches = job_tech_skills.intersection(cand_tech_skills)
            cluster_credits = 0.0
            for _, cluster_set in SKILL_CLUSTERS.items():
                missing_job = [s for s in job_tech_skills if s in cluster_set and s not in cand_tech_skills]
                cand_in_cl = [s for s in cand_tech_skills if s in cluster_set and s not in job_tech_skills]
                cluster_credits += min(len(missing_job), len(cand_in_cl)) * 0.5
            skill_affinity = (len(exact_matches) * 1.0 + cluster_credits) / len(job_tech_skills)

        jac_score = compute_token_overlap(cleaned_job, cleaned_resumes[idx])

        # Hybrid Blended Ensemble (45% SBERT + 35% TF-IDF + 15% Skill Taxonomy + 5% Jaccard)
        raw_blended = (
            (sbert_scores[idx] * 0.45) + 
            (cosine_sim_tfidf[idx] * 0.35) + 
            (skill_affinity * 0.15) + 
            (jac_score * 0.05)
        )

        # Baseline Floor Rescaling (removes universal ~0.22 English background overlap)
        BACKGROUND_FLOOR = 0.22
        calibrated_score = max(0.0, (raw_blended - BACKGROUND_FLOOR) / (1.0 - BACKGROUND_FLOOR))
        percentage = calibrated_score * 100.0

        # Dynamic Experience Bonus (square-root scaling prevents inflating weak matches)
        cand_exp = parse_work_tenure(resume_text)
        if required_experience > 0 and cand_exp >= required_experience:
            relevance_scale = min(1.0, np.sqrt(calibrated_score))
            percentage += 15.0 * relevance_scale

        # Cross-Domain Career Transferability Matrix
        for cand_pat, job_pat, boost_val in CAREER_TRANSFERS:
            if cand_pat.search(resume_text) and job_pat.search(job_description):
                percentage += boost_val
                break

        # Technical Gatekeeper: Hard zero for profiles lacking any identified technical skills
        if not cand_tech_skills:
            percentage = 0.0

        percentage = round(min(percentage, 100.0), 2)

        # Decision Thresholds
        if percentage >= 65.0:
            result_label = "Highly Suitable"
        elif percentage >= 45.0:
            result_label = "Suitable"
        else:
            result_label = "Low Match"

        evaluated_candidates.append({
            'candidate_id': f"Candidate {idx + 1}",
            'score': percentage,
            'label': result_label
        })

    evaluated_candidates.sort(key=lambda item: item['score'], reverse=True)
    return evaluated_candidates