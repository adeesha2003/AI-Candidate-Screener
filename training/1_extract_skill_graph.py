import json
import os
import random
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==========================================
# CONFIGURATION
# ==========================================
# Point this to your local Qwen model's API URL (Ollama or LM Studio)
LOCAL_API_BASE = "http://localhost:11434/v1" # Ollama default
# LOCAL_API_BASE = "http://localhost:1234/v1" # LM Studio default
LOCAL_MODEL_NAME = "qwen3:8b" # Change this to your exact local model name

DATASET_PATH = r"C:\Users\Kavinda\Desktop\sem 4\AI\group project ai\Job_Descriptions\job_dataset.json"
OUTPUT_PATH = "skill_graph.json"

def create_mock_skill_graph():
    """Generates a sample skill graph."""
    print("Generating a sample skill graph for demonstration...")
    mock_data = [
        {"text_a": "Python", "text_b": "Django", "label": 1.0, "category": "tech"},
        {"text_a": "Machine Learning", "text_b": "TensorFlow", "label": 1.0, "category": "tech"},
        {"text_a": "React", "text_b": "JavaScript", "label": 1.0, "category": "tech"},
        {"text_a": "Marketing Strategy", "text_b": "SEO Optimization", "label": 1.0, "category": "marketing"},
        {"text_a": "Recruitment", "text_b": "Talent Acquisition", "label": 1.0, "category": "hr"},
        {"text_a": "Python", "text_b": "Marketing Strategy", "label": 0.0, "category": "mixed"},
        {"text_a": "React", "text_b": "Talent Acquisition", "label": 0.0, "category": "mixed"}
    ]
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(mock_data, f, indent=4)
    print(f"Saved mock data to {OUTPUT_PATH}")

def extract_skills_with_local_qwen():
    """Uses your local Qwen model via OpenAI API format to extract word relations."""
    client = OpenAI(base_url=LOCAL_API_BASE, api_key="lm-studio") # api_key is required by the client but ignored locally
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    sample_jds = random.sample(dataset, min(5, len(dataset)))
    text_corpus = ""
    for jd in sample_jds:
        title = jd.get('Title', '')
        skills = ", ".join(jd.get('Skills', []))
        resp = ", ".join(jd.get('Responsibilities', []))
        text_corpus += f"Title: {title}\nSkills: {skills}\nResponsibilities: {resp}\n\n"
    
    prompt = f"""
    Act as an expert HR and Machine Learning taxonomist.
    Analyze the following job descriptions and extract key skills, tools, and domain keywords.
    Group related words together (e.g., tech tools together, marketing terms together).
    
    Return a raw JSON list of word pairs with a similarity score (1.0 for related, 0.0 for strictly unrelated).
    Format must be exactly like this, no markdown formatting, no code blocks, just raw JSON text:
    [
      {{"text_a": "Skill 1", "text_b": "Related Skill 2", "label": 1.0, "category": "domain"}},
      {{"text_a": "Skill 1", "text_b": "Unrelated Skill 3", "label": 0.0, "category": "mixed"}}
    ]
    
    Job Descriptions:
    {text_corpus[:3000]}
    """
    
    print(f"Sending data to local Qwen at {LOCAL_API_BASE}...")
    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        raw_output = response.choices[0].message.content
        clean_text = raw_output.strip().removeprefix('```json').removesuffix('```').strip()
        skill_graph = json.loads(clean_text)
        
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(skill_graph, f, indent=4)
        print(f"Successfully extracted {len(skill_graph)} skill relations and saved to {OUTPUT_PATH}")
        
    except Exception as e:
        print("Failed to communicate with local model or parse JSON.")
        print("Error:", e)
        print("Is your local model server running on that port?")

if __name__ == "__main__":
    if OpenAI is None:
        print("OpenAI python package missing. Falling back to mock graph.")
        create_mock_skill_graph()
    else:
        extract_skills_with_local_qwen()
