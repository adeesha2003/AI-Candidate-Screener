import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

SKILL_GRAPH_PATH = "skill_graph.json"
WARMED_UP_MODEL_DIR = "./warmed-up-hr-model"
BASE_MODEL_NAME = "all-MiniLM-L6-v2"

def load_skill_graph(file_path):
    """Loads the skill graph and converts it to InputExamples for the model."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run 1_extract_skill_graph.py first.")
        
    train_examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            # We train the model to understand the similarity (label) between text_a and text_b
            example = InputExample(
                texts=[str(item['text_a']), str(item['text_b'])], 
                label=float(item['label'])
            )
            train_examples.append(example)
            
    return train_examples

def main():
    print(f"1. Loading skill relationships from {SKILL_GRAPH_PATH}...")
    try:
        train_examples = load_skill_graph(SKILL_GRAPH_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Found {len(train_examples)} skill pairs.")

    print(f"2. Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # Prepare DataLoader
    # We use a small batch size because the skill graph dataset is usually small
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    
    # CosineSimilarityLoss forces the model to push vectors closer for label 1.0 (Python/Django) 
    # and push them apart for label 0.0 (Python/Marketing)
    train_loss = losses.CosineSimilarityLoss(model=model)

    print("3. Starting WARMUP Training...")
    # Because these are just short words/skills, we can run a few more epochs very quickly
    num_epochs = 10 
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=WARMED_UP_MODEL_DIR,
        show_progress_bar=True
    )

    print(f"\nWarmup complete! The model now understands domain word relationships.")
    print(f"Warmed-up model saved to: {WARMED_UP_MODEL_DIR}")
    print("You can now proceed to run 3_final_train_model.py")

if __name__ == "__main__":
    main()
