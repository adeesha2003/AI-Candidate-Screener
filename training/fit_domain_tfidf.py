import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
CSV_RESUMES_FILE = "UpdatedResumeDataSet.csv"
JSON_RESUMES_FILE = "raw_text_1000.json"
JD_FILE = os.path.join("Job_Descriptions", "job_dataset.json")

MODEL_OUTPUT_ROOT = "domain_tfidf_model.pkl"
MODEL_OUTPUT_WEB = os.path.join("web_app", "domain_tfidf_model.pkl")

# Custom HR / Resume noise words to exclude
HR_STOPWORDS = {
    'curriculum', 'vitae', 'resume', 'responsibilities', 'duties',
    'references', 'available', 'request', 'seeking', 'opportunity',
    'passionate', 'hardworking', 'team', 'player', 'etc'
}

def load_all_documents():
    corpus = []

    # 1. Load real resumes from UpdatedResumeDataSet.csv
    if os.path.exists(CSV_RESUMES_FILE):
        print(f"[1/3] Loading real resumes from {CSV_RESUMES_FILE}...")
        df = pd.read_csv(CSV_RESUMES_FILE)
        if "Resume" in df.columns:
            resumes = df["Resume"].dropna().tolist()
            corpus.extend(resumes)
            print(f"      -> Loaded {len(resumes)} real resumes from CSV.")
    else:
        print(f"[1/3] {CSV_RESUMES_FILE} not found, skipping.")

    # 2. Load additional resumes from raw_text_1000.json (if available)
    if os.path.exists(JSON_RESUMES_FILE):
        print(f"[2/3] Loading additional resumes from {JSON_RESUMES_FILE}...")
        with open(JSON_RESUMES_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            json_resumes = [item.get("raw_text", "").strip() for item in raw_data if item.get("raw_text", "").strip()]
            corpus.extend(json_resumes)
            print(f"      -> Loaded {len(json_resumes)} resumes from JSON.")
    else:
        print(f"[2/3] {JSON_RESUMES_FILE} not found, skipping.")

    # 3. Load Job Descriptions from job_dataset.json
    if os.path.exists(JD_FILE):
        print(f"[3/3] Loading job descriptions from {JD_FILE}...")
        with open(JD_FILE, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
            for j in jobs:
                title = j.get("Title", "")
                skills = " ".join(j.get("Skills", [])) if isinstance(j.get("Skills"), list) else str(j.get("Skills", ""))
                resp = " ".join(j.get("Responsibilities", [])) if isinstance(j.get("Responsibilities"), list) else str(j.get("Responsibilities", ""))
                jd_text = f"{title}. Skills: {skills}. Responsibilities: {resp}"
                if jd_text.strip():
                    corpus.append(jd_text)
            print(f"      -> Loaded {len(jobs)} job descriptions.")
    else:
        print(f"[3/3] {JD_FILE} not found, skipping.")

    print(f"\nTotal recruitment documents collected: {len(corpus)}")
    return corpus

def main():
    corpus = load_all_documents()
    if not corpus:
        print("Error: No documents found to train on.")
        return

    print("\nFitting Domain-Specific TfidfVectorizer...")
    
    # Configure vectorizer with best practices for resume matching
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Single words & bigrams (e.g. 'machine learning', 'sql server')
        sublinear_tf=True,         # Sublinear TF: 1 + log(tf) to prevent keyword stuffing
        min_df=2,                  # Ignore terms that appear only once (names, phone numbers, typos)
        max_df=0.85,               # Ignore terms appearing in >85% of documents (too generic)
        stop_words='english'
    )

    vectorizer.fit(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Successfully fitted! Learned vocabulary size: {vocab_size:,} terms & phrases.")

    # Save to project root
    joblib.dump(vectorizer, MODEL_OUTPUT_ROOT)
    print(f"Saved model to: {MODEL_OUTPUT_ROOT}")

    # Save to web_app folder if it exists
    if os.path.exists("web_app"):
        joblib.dump(vectorizer, MODEL_OUTPUT_WEB)
        print(f"Saved model copy to: {MODEL_OUTPUT_WEB}")

    print("\n--- Domain TF-IDF Training Complete! ---")

if __name__ == "__main__":
    main()
