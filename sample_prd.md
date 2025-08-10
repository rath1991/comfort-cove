
---

# **Product Requirements Document – Failure Mode Prediction & ISO Mapping Tool**

## **1. Overview**

This product automates classification of work orders into SME-aligned failure modes and optionally maps them to ISO 14224 failure mode codes. It leverages a **three-stage LLM pipeline** (already implemented) combined with **filtered vector search** for ISO mapping. The end goal is to **accelerate SME productivity** by pre-filling failure modes and providing transparent, editable predictions.

---

## **2. Objectives**

* **Primary:** Classify failure modes from work order text (Short, Long, Full) in a single overnight batch run.
* **Secondary:** Map classified failure modes to ISO 14224 codes using a two-stage LLM + filtered retrieval process.
* **Tertiary:** Provide a **Streamlit frontend** for SMEs to:

  * Review predictions
  * Accept, edit, or reject
  * View justifications
  * See ISO mapping (on demand)
  * Save changes and export

---

## **3. Functional Requirements**

### **3.1 Batch Processing**

* Input: Excel/CSV or Google Sheet with columns:

  * `wo_id`
  * `short_desc`
  * `long_desc`
  * `full_text` (concatenated or all fields combined)
* Process:

  1. **Stage 1 LLM**: Predict SME-aligned failure mode + justification + normalized text
  2. **Stage 2–4 LLM** (ISO mapping):

     * Classify main equipment category
     * Classify equipment class(es)
     * Filter vector DB search to category + classes
     * Retrieve top matches
     * Select best ISO code with justification
* Output: Enriched dataset with:

  * `predicted_failure_mode`
  * `prediction_confidence`
  * `prediction_justification`
  * `iso_code`
  * `iso_confidence`
  * `iso_justification`

---

### **3.2 Frontend (Streamlit)**

#### **Main Features**

1. **Dataset Upload & Mapping**

   * Upload Excel/CSV or connect Google Sheet
   * Map columns (Short, Long, Full, etc.)

2. **Review Workspace**

   * Table view with filters:

     * Confidence threshold
     * Category / Class
     * Needs review vs accepted
   * Row card:

     * WO Short & Long text
     * Predicted failure mode + confidence
     * Expandable justification
     * ISO mapping fields (collapsed by default)
     * Actions: Accept / Edit / Needs SME

3. **Bulk Actions**

   * Auto-accept all above confidence threshold
   * Apply one decision to multiple rows

4. **Export**

   * Write back to same Google Sheet (new columns appended)
   * CSV download with all metadata

5. **File Access**

   * "View file" button to open source Excel/Sheet in a side panel or external tab

---

### **3.3 Backend Logic**

* **Batch Mode**:

  * Overnight cron/scheduled job runs LLM pipeline on all new rows
  * Caches results by hashing `(full_text + prompt_version)` to avoid recomputation
* **On-demand ISO Mapping**:

  * Only runs when SME clicks "Map to ISO" if not already populated
* **Confidence Handling**:

  * Auto-apply ≥ threshold
  * Queue mid-confidence for SME
  * Flag low-confidence for full review

---

### **3.4 Data Model**

| Field                     | Type     | Notes                            |
| ------------------------- | -------- | -------------------------------- |
| wo\_id                    | string   | Work order ID                    |
| short\_desc               | string   | Short description                |
| long\_desc                | string   | Long description                 |
| full\_text                | string   | Combined text                    |
| predicted\_failure\_mode  | string   | LLM output                       |
| prediction\_confidence    | float    | 0–1                              |
| prediction\_justification | string   | LLM rationale                    |
| iso\_code                 | string   | ISO 14224 code                   |
| iso\_confidence           | float    | Confidence from retrieval or LLM |
| iso\_justification        | string   | LLM rationale                    |
| final\_failure\_mode      | string   | SME-reviewed                     |
| final\_iso\_code          | string   | SME-reviewed                     |
| action                    | string   | accept/edit/reject               |
| reviewer                  | string   | SME name/ID                      |
| timestamp                 | datetime |                                  |

---

## **4. Non-Functional Requirements**

* **Performance:** Batch run should handle ≥ 10k rows overnight
* **Traceability:** Store prompt version & few-shot set used for each prediction
* **Auditability:** Every export includes run ID and prompt hash
* **Extensibility:** Easy to swap embedding model or vector DB
* **Security:** Handle all SME review data locally or within approved infrastructure

---

## **5. Tech Stack**

* **LLM:** OpenAI GPT-4o-mini (classification, ISO mapping)
* **Embeddings:** OpenAI `text-embedding-3-large`
* **Vector DB (local):** Chroma
* **Frontend:** Streamlit
* **Backend:** Python, Pandas
* **Storage:** JSONL for ISO table, CSV/Excel for WO data
* **Deployment:** Local or containerized (Docker)

6. Constraints & Existing Components
Three-stage LLM pipeline for ISO mapping is complete and must not be modified.
Stage 1: Classify main equipment category from WO text.
Stage 2: Classify equipment class(es) for that category.
Stage 3: Filter vector DB by category + class, retrieve, and select ISO code with justification.
The pipeline is already implemented in pipeline.py and tested with ChromaDB.
All new product code (frontend, batch orchestration, SME review logic) will call this pipeline as a black-box function.
Integration points:
Input: wo_text (full, short, long combined)
Output: dict with main_category, equipment_classes, iso_code, iso_justification.
No refactoring or prompt changes inside the three-stage pipeline unless explicitly approved by SMEs.
