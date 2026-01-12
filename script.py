import pandas as pd
import ollama
import os
import re
import time
import pathway as pw
from tqdm import tqdm

DATA_DIR = "./data/novels/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def load_novel_with_pathway(file_path):
    table = pw.io.fs.read(
        file_path,
        format="text",
        autocommit_duration_ms=50
    )
    return table

def get_training_examples():
    """Loads real examples from train.csv to 'teach' Ollama the patterns."""
    try:
        df = pd.read_csv(TRAIN_FILE)
        ex_0 = df[df['label'] == 0].iloc[0]['content']
        ex_1 = df[df['label'] == 1].iloc[0]['content']
        return ex_0, ex_1
    except:
        return "Example: Faria was free in 1815. (Label 0)", "Example: Thalcave liked horses. (Label 1)"

def get_best_context(backstory, data_dir):
    is_monte = any(n in backstory for n in ["Dantes", "Noirtier", "Villefort", "Faria"])
    target = "The Count of Monte Cristo.txt" if is_monte else "In search of the castaways.txt"
    path = os.path.join(data_dir, target)
    
    if not os.path.exists(path): return "Text not found."
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    keywords = re.findall(r'\b[A-Z][a-z]+\b|\b\d{2,4}\b', backstory)
    
    paragraphs = content.split('\n\n')
    scored = []
    for p in paragraphs:
        score = sum(1 for kw in keywords if kw.lower() in p.lower())
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n---\n\n".join([p[1] for p in scored[:3]])

def judge_and_verify(backstory, context, ex0, ex1):
    """Ollama performs a two-pass audit."""
    
    initial_prompt = f"""
    TRAINING EXAMPLES:
    CONTRADICTION: {ex0} -> Label: 0
    CONSISTENT: {ex1} -> Label: 1
    
    TASK: Audit this BACKSTORY against the REFERENCE TEXT.
    REFERENCE TEXT: {context}
    BACKSTORY: {backstory}
    
    Identify if it is a lie (0) or plausible addition (1).
    Explain logic, then provide Label.
    """
    
    response1 = ollama.generate(model='llama3', prompt=initial_prompt, options={'temperature': 0})
    initial_audit = response1['response']

    verification_prompt = f"""
    ### SYSTEM ROLE: FORENSIC LITERARY JUDGE ###
    You are a strict fact-checker. Your mission is to detect "Fan-Fiction" additions.
    
    INITIAL AUDIT DATA: {initial_audit}
    BACKSTORY TO EVALUATE: {backstory}

    ### STRIKE SYSTEM (ANY STRIKE = LABEL 0) ###
    1. IDENTITY STRIKE: Nationality, parents, or birthplace changes (e.g., a French character born in Britain).
    2. SKILL STRIKE: New "expert" skills or prestigious jobs not in the book (e.g., "Toxicology", "Admiralty Cartographer", "Junior Warrior").
    3. SEVERITY STRIKE: Adds deaths, major crimes, or tragic trauma not in the original plot.
    4. UNRELATED STRIKE: The story feels like a fable or a completely different narrative (e.g., the "Shark" backstory).

    ### THE GOLDEN RULE ###
    If the backstory adds 'Complexity' (tragic past, secret motives, secret wealth) that the reference text doesn't mention, it is a LIE. Label: 0. 
    Accept Label: 1 ONLY if the detail is minor, harmless, and fits the character's existing job/tone perfectly.

    ### MANDATORY OUTPUT FORMAT ###
    Label: [0 or 1]
    Rationale: [State which Strike Rule was broken. If Label 1, state why it is a harmless canonical fit.]
    
    CRITICAL: If Rationale mentions "not mentioned" or "not supported", Label MUST be 0.
    """
    
    response2 = ollama.generate(model='llama3', prompt=verification_prompt, options={'temperature': 0})
    return response2['response']

def run_pipeline():
    ex0, ex1 = get_training_examples()
    test_df = pd.read_csv(TEST_FILE)
    results = []
    
    print(f"🚀 Starting Verification...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        context = get_best_context(row['content'], DATA_DIR)
        verdict = judge_and_verify(row['content'], context, ex0, ex1)
        label = "0" if "Label: 0" in verdict else "1"
        rationale = verdict.split("Rationale:")[-1].strip() if "Rationale:" in verdict else "Verified locally."
        results.append({'Story ID': row['id'], 'Prediction': label, 'Rationale': rationale})
        pd.DataFrame(results).to_csv("results.csv", index=False)
    print(f"🚀Check results.csv...")

if __name__ == "__main__":
    run_pipeline()