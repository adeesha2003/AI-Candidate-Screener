import os
import random
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
RESUME_FILE = "UpdatedResumeDataSet.csv"
JD_FILE = os.path.join("Job_Descriptions", "job_dataset.csv")
OUTPUT_MODEL_DIR = "./final-hr-model"
WEB_APP_MODEL_DIR = os.path.join("web_app", "final-hr-model")
BASE_MODEL_NAME = "all-MiniLM-L6-v2"

# Domain mapping between Resume categories and Job titles/keywords
DOMAIN_MAP = {
    'Python Developer': ['python', 'backend', 'django', 'flask', 'fastapi'],
    'Data Science': ['data science', 'machine learning', 'ai', 'data analyst', 'deep learning'],
    'Java Developer': ['java', 'spring', 'android', 'backend'],
    'DotNet Developer': ['.net', 'c#', 'asp.net', 'mvc'],
    'Web Designing': ['web', 'frontend', 'ui', 'ux', 'html', 'css', 'javascript', 'react'],
    'DevOps Engineer': ['devops', 'cloud', 'aws', 'docker', 'kubernetes', 'ci/cd', 'azure'],
    'Database': ['database', 'sql', 'dba', 'oracle', 'postgres'],
    'Hadoop': ['hadoop', 'big data', 'spark', 'etl'],
    'ETL Developer': ['etl', 'data warehouse', 'bi', 'business intelligence'],
    'Network Security Engineer': ['security', 'network', 'cyber', 'firewall'],
    'Testing': ['test', 'qa', 'quality assurance'],
    'Automation Testing': ['automation', 'selenium', 'testing', 'qa'],
    'Blockchain': ['blockchain', 'crypto', 'smart contract', 'solidity'],
    'Business Analyst': ['business analyst', 'product', 'requirements', 'agile'],
    'HR': ['hr', 'recruiter', 'human resources', 'talent', 'payroll']
}

def create_training_pairs():
    print("[1/4] Loading datasets...")
    if not os.path.exists(RESUME_FILE) or not os.path.exists(JD_FILE):
        raise FileNotFoundError("Missing resume or job dataset files.")

    df_resumes = pd.read_csv(RESUME_FILE)
    df_jds = pd.read_csv(JD_FILE)

    # Format job text
    jd_records = []
    for _, row in df_jds.iterrows():
        title = str(row.get('Title', ''))
        skills = str(row.get('Skills', ''))
        resp = str(row.get('Responsibilities', ''))
        text = f"Job Title: {title}. Required Skills: {skills}. Responsibilities: {resp}"
        jd_records.append({'title': title.lower(), 'text': text})

    # Group resumes by category
    category_resumes = {}
    for _, row in df_resumes.iterrows():
        cat = row['Category']
        text = str(row['Resume'])
        if cat not in category_resumes:
            category_resumes[cat] = []
        category_resumes[cat].append(text)

    print(f"      Resumes categorized into {len(category_resumes)} fields.")
    print(f"      Total available Job Descriptions: {len(jd_records)}")

    # Generate balanced pairs
    examples = []
    
    # 1. Positive Pairs (Score: 0.85 - 0.95)
    print("[2/4] Generating positive, partial, and negative training pairs...")
    for cat, keywords in DOMAIN_MAP.items():
        matching_jds = [j for j in jd_records if any(k in j['title'] for k in keywords)]
        if not matching_jds or cat not in category_resumes:
            continue
        
        for resume in category_resumes[cat]:
            jd = random.choice(matching_jds)
            label = round(random.uniform(0.85, 0.95), 2)
            examples.append(InputExample(texts=[jd['text'], resume], label=label))

    # 2. Partial Match Pairs (Score: 0.40 - 0.55)
    partial_cross = [
        ('Python Developer', 'Data Science'),
        ('Java Developer', 'DotNet Developer'),
        ('Web Designing', 'Java Developer'),
        ('Database', 'Data Science'),
        ('Automation Testing', 'Java Developer'),
        ('DevOps Engineer', 'Network Security Engineer')
    ]
    for cat1, cat2 in partial_cross:
        keywords = DOMAIN_MAP.get(cat2, [])
        target_jds = [j for j in jd_records if any(k in j['title'] for k in keywords)]
        if cat1 in category_resumes and target_jds:
            sample_resumes = random.sample(category_resumes[cat1], min(25, len(category_resumes[cat1])))
            for resume in sample_resumes:
                jd = random.choice(target_jds)
                label = round(random.uniform(0.40, 0.55), 2)
                examples.append(InputExample(texts=[jd['text'], resume], label=label))

    # 3. Negative / Unrelated Pairs (Score: 0.05 - 0.15)
    unrelated_categories = ['Advocate', 'Arts', 'Mechanical Engineer', 'Civil Engineer', 'Sales', 'Health and fitness']
    tech_jds = [j for j in jd_records if any(k in j['title'] for k in ['python', 'developer', 'engineer', 'data', 'cloud'])]
    
    for un_cat in unrelated_categories:
        if un_cat in category_resumes and tech_jds:
            for resume in category_resumes[un_cat]:
                jd = random.choice(tech_jds)
                label = round(random.uniform(0.05, 0.15), 2)
                examples.append(InputExample(texts=[jd['text'], resume], label=label))

    random.shuffle(examples)
    print(f"      Total generated training pairs: {len(examples)}")
    return examples

def main():
    examples = create_training_pairs()
    
    # 85% Train, 15% Evaluation
    split_idx = int(len(examples) * 0.85)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"      Training pairs: {len(train_examples)} | Validation pairs: {len(val_examples)}")

    # Check CUDA GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] Initializing base model '{BASE_MODEL_NAME}' on {device.upper()}...")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(BASE_MODEL_NAME, device=device)

    # Setup Dataloader and Loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32 if device == "cuda" else 16)
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Setup Evaluator to track improvement
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='hr-val')

    print("[4/4] Fine-tuning Sentence-BERT on Domain Recruitment Data...")
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=50,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_MODEL_DIR,
        show_progress_bar=True
    )

    print(f"\nModel fine-tuned successfully and saved to: {OUTPUT_MODEL_DIR}")
    
    # Also save to web_app directory so web app can use it directly
    if os.path.exists("web_app"):
        model.save(WEB_APP_MODEL_DIR)
        print(f"Model copy saved to: {WEB_APP_MODEL_DIR}")

    print("\n=== SBERT Fine-Tuning Complete! ===")

if __name__ == "__main__":
    main()
