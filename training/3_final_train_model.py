import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ==========================================
# CONFIGURATION
# ==========================================
# Pointing to the actual dataset directory you provided
DATASET_DIR = r"C:\Users\Kavinda\Desktop\sem 4\AI\group project ai\Job_Descriptions"
DATASET_FILE = os.path.join(DATASET_DIR, "job_dataset.json") # or job_dataset.csv

# We are NO LONGER using the base model. 
# We are loading the model that was just taught the word relationships!
WARMED_UP_MODEL_DIR = './warmed-up-hr-model'
FINAL_MODEL_DIR = './final-hr-model'

def load_data(file_path):
    """
    Loads JD/Resume dataset and converts to InputExample format.
    Supports both CSV and JSON formats depending on what is in your directory.
    """
    train_examples = []
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            # Assuming columns: job_description, resume, similarity_score
            example = InputExample(
                texts=[str(row.get('job_description', '')), str(row.get('resume', ''))], 
                label=float(row.get('similarity_score', 0.5)) # fallback to 0.5 if missing
            )
            train_examples.append(example)
            
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                example = InputExample(
                    texts=[str(item.get('job_description', '')), str(item.get('resume', ''))], 
                    label=float(item.get('similarity_score', 0.5))
                )
                train_examples.append(example)
    else:
        raise ValueError(f"Unsupported file type for {file_path}. Use .csv or .json")
        
    return train_examples


def main():
    print(f"1. Loading full document dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: {DATASET_FILE} not found. Please ensure the filename is correct.")
        return
        
    try:
        train_examples = load_data(DATASET_FILE)
        print(f"Successfully loaded {len(train_examples)} document pairs.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"2. Loading the WARMED-UP model from {WARMED_UP_MODEL_DIR}...")
    if not os.path.exists(WARMED_UP_MODEL_DIR):
        print(f"ERROR: {WARMED_UP_MODEL_DIR} not found. You must run 2_warmup_model.py first!")
        return
        
    # Load the model we prepared in Stage 1
    model = SentenceTransformer(WARMED_UP_MODEL_DIR)
    
    # 3. Prepare Training
    train_batch_size = 16
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    print("4. Starting Final Document Fine-Tuning...")
    num_epochs = 4
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=FINAL_MODEL_DIR,
        show_progress_bar=True
    )
    
    print(f"\nSUCCESS! The ultimate, domain-adapted model has been saved to {FINAL_MODEL_DIR}")
    print("You can now use it in your Screening System by calling:")
    print("model = SentenceTransformer('./final-hr-model')")

if __name__ == '__main__':
    main()
