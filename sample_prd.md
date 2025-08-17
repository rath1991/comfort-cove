
# **Product Requirements Document – RCA & Fault Tree Generation**

## **1. Overview**

This feature extends the failure mode classification + ISO 14224 mapping pipeline into a **root cause analysis (RCA) system**.
The system ingests chronological work order text, extracts evidence, generates **mini fault trees using an LLM**, scores hypotheses against evidence, and produces **root cause explanations + preventive recommendations**.
It delivers both a **Streamlit UI for SMEs** and a **FastAPI backend** for programmatic access.

---

## **2. Objectives**

* Ingest free-text work orders and transform them into structured evidence.
* Generate bounded fault trees and hypotheses with reasoning LLMs.
* Select most probable root cause(s) with transparent evidence citations.
* Recommend practical actions and risk warnings.
* Provide **Streamlit frontend** and **FastAPI backend endpoints** for seamless SME use and integration.

---

## **3. Functional Requirements**

### **3.1 Data Inputs**

* **Work Orders dataset** (Excel/CSV/Google Sheet).
  Fields: `wo_id`, `equipment_id`, `date`, `short_desc`, `long_desc`, `full_text`.
* **Pipeline outputs (from 3-stage classifier):** `predicted_failure_mode`, `iso_code` (optional).
* **Optional telemetry features:** vibration, temperature, current, trips.

---

### **3.2 Pipeline Steps**

1. **Evidence Extraction (LLM)**

   * Extract structured facts from work order text.
   * Fields: `id`, `date`, `symptoms[]`, `actions[]`, `outcome`, `quotes[]`.

2. **Hypothesis Generation (LLM)**

   * Generate 3–5 candidate causes from a fixed **Cause Taxonomy**.
   * Each with rationale, expected support, expected contradiction.

3. **Fault Tree Drafting (LLM)**

   * Build a shallow tree (max depth=2, breadth=4).
   * Nodes reference taxonomy causes + evidence IDs.
   * Mark unsupported nodes as `assumed:true`.

4. **Hypothesis Scoring**

   * Score each hypothesis (0–1) using evidence.
   * Support vs contradiction vs assumptions.

5. **Root Cause Selection**

   * Pick top hypothesis or mark “insufficient evidence.”

6. **Recommendation Generation**

   * 2–3 prioritized actions with rationale, expected impact, urgency.

---

### **3.3 FastAPI Backend**

Create a separate `api.py` file with endpoints:

* `POST /evidence` → Extract structured evidence from WO text.
* `POST /hypotheses` → Generate hypotheses + fault tree from evidence.
* `POST /score` → Score hypotheses against evidence.
* `POST /rca` → Full pipeline: WO text → evidence → RCA JSON (root cause + recs).
* `GET /equipment/{id}` → Retrieve RCA timeline & latest root cause for equipment.

Responses must be in structured JSON (schema consistent with pipeline).

---

### **3.4 Streamlit Frontend**

* **Dashboard**

  * Equipment list with RCA status.
  * Filters: equipment type, failure mode, risk level.

* **RCA Detail Page**

  * Timeline of failures (Plotly chart).
  * RCA card: root cause, confidence, recommendations.
  * Expandable fault tree preview.
  * “Why?” button → evidence table with quotes.

* **Review Tools**

  * Accept / Edit / Reject RCA.
  * Save SME overrides back to dataset.

* **Recommendations Board**

  * Aggregated preventive actions across equipment.
  * Filters: urgency, equipment type, cause family.

---

## **4. Data Model**

| Field           | Type     | Notes               |
| --------------- | -------- | ------------------- |
| equipment\_id   | string   | Asset ID            |
| wo\_id          | string   | Work order ID       |
| date            | date     | WO date             |
| evidence        | JSON     | Extracted per WO    |
| hypotheses      | JSON\[]  | Candidate causes    |
| fault\_tree     | JSON     | Mini tree           |
| root\_cause     | JSON     | Selected hypothesis |
| recommendations | JSON\[]  | Preventive actions  |
| confidence      | float    | 0–1                 |
| citations       | JSON\[]  | WO IDs + quotes     |
| sme\_override   | string   | SME RCA override    |
| reviewer        | string   | SME name            |
| timestamp       | datetime | Review time         |

---

## **5. Non-Functional Requirements**

* Must process \~10k WOs in batch.
* Every RCA must include **citations to WOs**.
* Log all prompts and prompt versions for audit.
* Expose structured JSON for downstream analytics.
* LLM must never assert causes without marking them as `assumed`.

---

## **6. Tech Stack**

* **LLM:** OpenAI GPT-4o-mini or GPT-4 for reasoning.
* **Embeddings (optional):** OpenAI `text-embedding-3-large`.
* **Vector DB:** Chroma (for retrieval, optional).
* **Backend:** FastAPI, Python.
* **Frontend:** Streamlit, Plotly.
* **Storage:** Pandas/CSV or SQLite for persistence.

---

## **7. Prompts (Core)**

### **Evidence Extraction**

```
Extract only facts from WORK ORDERS into JSON:
{ id, date, symptoms[], actions[], outcome, quotes[] }.
Do not hypothesize.
```

### **Hypothesis Generation**

```
Using EVIDENCE + TAXONOMY [list], propose 3–5 hypotheses. 
Each with rationale, expect_support[], expect_contra[]. 
If insufficient evidence, say so.
```

### **Fault Tree Draft**

```
Build a small fault tree for TOP EVENT = <summary>. 
Only use causes from TAXONOMY. 
Max depth=2, breadth=4. 
Mark unsupported nodes as assumed:true.
```

### **Scoring & Root Cause**

```
Score each hypothesis 0..1 using ONLY EVIDENCE. 
Return support_ids[], contra_ids[], score, notes. 
Pick the top hypothesis or mark 'insufficient evidence'.
```

### **Recommendations**

```
From ROOT CAUSE, propose 2–3 prioritized actions. 
Each: action, why, expected impact, priority, window_days.
```

---

## **8. Constraints**

* The existing **3-stage ISO classification pipeline must remain unchanged**.
* RCA feature is **add-on only**.
* All claims must be evidence-bound; otherwise marked `assumed` or `insufficient`.

---

This PRD ensures GitHub Copilot Agent will:

* Build the **pipeline** (evidence → hypotheses → RCA).
* Scaffold a **FastAPI backend (api.py)** with endpoints.
* Build a **Streamlit frontend** for SME review.
