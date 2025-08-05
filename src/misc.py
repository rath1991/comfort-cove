# GitHub Copilot Prompt:
# 
# I need Python code for a complete two-stage LLM classification + filtered vector retrieval pipeline for ISO 14224 failure mode matching.
# Steps:
# 1. Load ISO failure mode data from a JSONL file (output from Azure Document Intelligence table extraction).
#    - Each line contains: failure_mode_code, description, examples (list), equipment_category, equipment_classes (list of codes).
# 2. Each ISO table row is treated as a single "chunk" for vector embeddings.
# 3. Create a canonical text for embeddings combining: failure_mode_code, description, examples, category, classes.
# 4. Use OpenAI "text-embedding-3-large" or similar to generate embeddings.
# 5. Store in a vector DB (Chroma example) with metadata: failure_mode_code, equipment_category, equipment_classes.
# 6. For a new work order (WO) query:
#    a. Pass WO text to LLM #1 to classify the main equipment category from a fixed list: Mechanical, Electrical, Instrumentation, Static.
#    b. Pass WO text + main category to LLM #2 to choose the most likely 1-2 equipment classes from a given list for that category.
#    c. Use vector DB similarity search filtered by main_category and predicted equipment_classes.
#    d. Retrieve top-k matching ISO entries.
#    e. Pass WO text + top matches to another LLM prompt to pick the most relevant failure_mode_code and return JSON with code and justification.
# 7. Make the code modular so I can swap out embedding models or vector DBs later.
# 8. Include an example `if __name__ == "__main__":` block showing the pipeline run end-to-end with a sample WO text.

import json
from typing import List, Dict
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = "your_api_key"
client = OpenAI(api_key=OPENAI_API_KEY)

# Vector DB setup (Chroma example)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="iso_failure_modes",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-large"
    )
)

# -----------------------
# STEP 1 - LOAD ISO DATA (already from Doc Intelligence extraction)
# -----------------------
def load_iso_json(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# -----------------------
# STEP 2 - CHUNKING (one row = one chunk)
# -----------------------
def prepare_embedding_text(row: Dict) -> str:
    return (
        f"Failure Mode Code: {row['failure_mode_code']}\n"
        f"Description: {row['description']}\n"
        f"Examples: {', '.join(row['examples'])}\n"
        f"Category: {row['equipment_category']}\n"
        f"Classes: {', '.join(row['equipment_classes'])}"
    )

# -----------------------
# STEP 3 - CREATE VECTOR INDEX
# -----------------------
def index_iso_data(iso_data: List[Dict]):
    for i, row in enumerate(iso_data):
        collection.add(
            ids=[f"row_{i}"],
            documents=[prepare_embedding_text(row)],
            metadatas=[{
                "failure_mode_code": row["failure_mode_code"],
                "equipment_category": row["equipment_category"],
                "equipment_classes": row["equipment_classes"]
            }]
        )
    print(f"Indexed {len(iso_data)} ISO entries.")

# -----------------------
# STEP 4 - LLM CLASSIFICATION
# -----------------------
def classify_main_category(wo_text: str) -> str:
    prompt = f"""
Given the following work order description:
\"\"\"{wo_text}\"\"\"
Pick the most relevant main equipment category from:
Mechanical, Electrical, Instrumentation, Static.
Return only the category name.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

def classify_equipment_class(wo_text: str, main_category: str, valid_classes: List[str]) -> List[str]:
    prompt = f"""
Work order: \"\"\"{wo_text}\"\"\"
Main category: {main_category}

Valid equipment classes for this category: {', '.join(valid_classes)}

Pick the top 1–2 most relevant equipment class codes from the list.
Return only the codes, comma separated.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return [c.strip() for c in resp.choices[0].message.content.split(",")]

# -----------------------
# STEP 5 - FILTERED RETRIEVAL
# -----------------------
def retrieve_failure_modes(wo_text: str, main_category: str, classes: List[str], top_k: int = 5):
    results = collection.query(
        query_texts=[wo_text],
        n_results=top_k,
        where={
            "equipment_category": main_category,
            "equipment_classes": {"$in": classes}
        }
    )
    return results

# -----------------------
# STEP 6 - FINAL SELECTION
# -----------------------
def select_best_code(wo_text: str, retrieved: Dict) -> Dict:
    top_matches = [
        f"{retrieved['metadatas'][0][i]['failure_mode_code']} - {retrieved['documents'][0][i]}"
        for i in range(len(retrieved['documents'][0]))
    ]
    prompt = f"""
Work order: \"\"\"{wo_text}\"\"\"

Top matching ISO failure modes:
{chr(10).join(top_matches)}

Pick the single most relevant failure mode code and explain briefly why.
Return JSON: {{"matched_code": "...", "justification": "..."}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content)

# -----------------------
# RUN PIPELINE
# -----------------------
if __name__ == "__main__":
    # 1. Load ISO data (already processed by Doc Intelligence)
    iso_data = load_iso_json("iso_failure_modes.jsonl")

    # 2. Index data into vector DB
    index_iso_data(iso_data)

    # Example query
    wo_text = "Pump leaking oil at base with abnormal vibration"

    # 3. Step 1 LLM → main category
    main_category = classify_main_category(wo_text)
    print("Main category:", main_category)

    # 4. Step 2 LLM → equipment class
    valid_classes_map = {
        "Mechanical": ["CE", "CO", "GT", "PU", "ST", "TE"],
        "Electrical": ["EG", "EM"],
        "Instrumentation": ["INSTR1", "INSTR2"],
        "Static": ["HX", "TK", "VS", "PI"]
    }
    predicted_classes = classify_equipment_class(wo_text, main_category, valid_classes_map[main_category])
    print("Predicted classes:", predicted_classes)

    # 5. Retrieval
    retrieved = retrieve_failure_modes(wo_text, main_category, predicted_classes)
    print("Retrieved entries:", retrieved)

    # 6. Final selection & justification
    final_output = select_best_code(wo_text, retrieved)
    print("Final matched code & justification:", final_output)
