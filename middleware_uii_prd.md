

# 📄 PRD 1: Middleware (FastAPI)

## Objective

Provide a **read-only API layer** to serve failure mode standardization and RCA data from JSON files so Streamlit can consume them without touching raw files.

---

## Data Sources

* **`failure_modes.json`** → master list of failure modes and metadata.
* **`standardization/wo_results.jsonl`** → standardized work order results (one JSON per WO).
* **`rca_results/<equipment_id>-rca.json`** → detailed RCA results for each equipment.
* **`analytics/*.json`** (optional precomputed stats/charts).

---

## Endpoints

### 1. Health

* **GET** `/health`
* Response:

```json
{ "status":"ok","ts":"2025-08-20T01:23:45Z" }
```

---

### 2. Failure Modes

* **GET** `/failure_modes`
* Loads `failure_modes.json`.
* Response:

```json
[
  {"code":"ELF","description":"External leakage - fuel","examples":["External leakage of supplied fuel/gas"],"category":"Mechanical","classes":["PU","CE"]},
  {"code":"BRD","description":"Breakdown","examples":["Seizure","Breakage"],"category":"Mechanical","classes":["GT","ST"]}
]
```

---

### 3. Work Orders (Standardization)

* **GET** `/standardization/workorders`
* Params: `page`, `page_size`, `date_from`, `date_to`, `equipment_id`, `class`, `confidence_min`, `search`
* Response:

```json
{
  "page":1,"page_size":50,"total":10324,
  "rows":[
    {"wo_id":"WO-803338","equipment_id":"WS1-0-AT-11121","date":"2025-06-12",
     "short_desc":"Cylinder 2 empty","long_desc":"Repeated empty condition...",
     "predicted_failure_mode":"Deferred maintenance","confidence":0.88,
     "justification":"Recurring cylinder empties suggest deferred maintenance",
     "iso_code":"ELF","iso_confidence":0.72}
  ]
}
```

---

### 4. Analytics

* **GET** `/standardization/stats`

* Response: counts + last run metadata.

* **GET** `/standardization/charts/failure_modes?top_n=10`

* Response: list of label/count.

* **GET** `/standardization/charts/weekly_trend`

* Response: standardized vs auto-applied per week.

* **GET** `/standardization/bad_actors?limit=50`

* Response: list of equipment with failures count, top mode, risk.

---

### 5. RCA

* **GET** `/rca/equipment/{equipment_id}`

* Loads `rca_results/<equipment_id>-rca.json`.

* Response schema includes:

  * `hypotheses[]`
  * `fault_tree`
  * `selected_root_cause`
  * `recommendations[]`

* **GET** `/rca/equipment/{equipment_id}/timeline`

* Slim view of failure events for plotting.

---

## Rules

* Middleware **only reads JSON**, no writes.
* If file not found → return `404`.
* Apply filters/pagination inside API, not in Streamlit.
* CORS open for Streamlit host.

---

# 📄 PRD 2: Streamlit Frontend

## Objective

Provide a **two-page Streamlit app**:

1. **Landing Page** — Failure Mode Standardization dashboard.
2. **RCA Page** — Detailed analysis for selected equipment.

---

## Landing Page

* **Header**

  * Title: “Maintenance Intelligence”
  * Last run, prompt version (from `/standardization/stats`)
* **KPI Cards**

  * Total WOs, % standardized, auto-applied, overrides
* **Charts**

  * Top failure modes (bar/pie → `/standardization/charts/failure_modes`)
  * Weekly trend (line/area → `/standardization/charts/weekly_trend`)
* **Bad Actors table**

  * From `/standardization/bad_actors`
  * Columns: equipment\_id, failures\_12m, top\_mode, risk, last\_failure
  * Action: **“Open RCA”** button
* **Work Orders Table**

  * Scrollable table of WOs (`/standardization/workorders`)
  * Columns: WO id, short desc, long desc (expand), predicted mode, confidence, justification
  * Optional: ISO code & confidence
* **Navigation**

  * “Open RCA” sets `st.session_state["equipment_id"]` and switches page.

---

## RCA Page

* **Header**

  * Equipment number + description
  * Timestamp of analysis
* **Root Cause Card**

  * Selected root cause
  * Confidence/score
  * Rationale
  * Expander for supporting/contradicting evidence
* **Hypotheses List**

  * ID, cause, score, assumptions
* **Fault Tree**

  * Tree diagram (or JSON preview if quick)
  * Evidence vs assumptions highlighted
* **Timeline**

  * Chronological failure events (from `/rca/equipment/{id}/timeline`)
* **Recommendations**

  * Action, urgency (Critical/High/Medium), window\_days, cost\_estimate, expected impact
  * Checkbox for accept/assign (local only, not persisted)

---

## Data Flow

* Streamlit calls middleware endpoints.
* All filters on frontend just append query params to API calls.
* Use `st.cache_data` for endpoint responses (ttl=60).

---

## Rules

* Streamlit must not open JSON files directly — only call API.
* Must handle empty arrays (show “No data available”).
* Use expanders for justification/rationale to keep UI uncluttered.

---

👉 With this separation:

* **Middleware PRD** = defines JSON APIs + contract.
* **Streamlit PRD** = defines UI + how it consumes API.

---

