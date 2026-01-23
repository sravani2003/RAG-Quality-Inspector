# 🔍 RAG Quality Inspector

A practical evaluation and inspection tool for **Retrieval-Augmented Generation (RAG)** systems that helps detect **hallucinations**, analyze **retrieval quality**, and visualize **model confidence**.

Built to answer a simple but critical question:

> *“Did the model actually use the retrieved context, or did it hallucinate?”*

---

## 🚀 What This Project Does

This app inspects the **end-to-end RAG pipeline** by:

- Retrieving relevant document chunks using **vector similarity**
- Generating answers using an LLM
- Measuring **retrieval confidence**
- Exposing **hallucination risk**
- Visually showing **which chunks influenced the answer**

---

## 🧠 Why This Matters

Most RAG demos stop at *“the answer looks correct”*.  
This tool goes further by answering:

- ❓ Was retrieval strong or weak?
- ❓ Is the answer grounded in retrieved context?
- ❓ When should the model abstain or warn?
- ❓ How confident is the system really?

This makes the system **inspectable, debuggable, and production-aware**.

---

## 🏗️ Architecture (High Level)

User Query
↓
Embedding Model
↓
Vector Database (Cosine Similarity)
↓
Top-K Retrieved Chunks
↓
LLM Answer Generation
↓
RAG Quality Inspection Layer


---

## 🔎 Key Features

### ✅ Retrieval Confidence Score
- Computed using cosine similarity of retrieved chunks
- Displayed as a numeric confidence score
- Used as a proxy to detect hallucination risk

### ✅ Hallucination Detection (Heuristic)
- Low retrieval confidence ⇒ High hallucination risk
- Threshold-based warning system
- Transparent and explainable (no black box)

### ✅ Retrieved Chunk Inspection
- Displays:
  - Chunk content
  - Similarity score
- Helps debug:
  - Bad chunking
  - Poor embeddings
  - Irrelevant retrieval

### ✅ Clean Streamlit UI
- Query input
- Confidence visualization
- Answer display
- Retrieval inspection panel

---

## 📊 Hallucination Logic (Explainable)

| Retrieval Confidence | Interpretation |
|----------------------|----------------|
| ≥ 0.65               | Strong grounding |
| 0.45 – 0.65          | Partial grounding |
| < 0.45               | High hallucination risk |

> These thresholds are configurable and can be tuned per dataset.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **Vector DB** – Embedding-based retrieval
- **LLMs (Groq / LLaMA)** – Answer generation
- **Cosine Similarity** – Retrieval scoring

---

## 🧪 Example Use Cases

- RAG system evaluation
- Debugging poor retrieval
- Comparing chunking strategies
- Detecting hallucinations before production
- Demonstrating RAG observability to stakeholders

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


📌 Future Improvements
Faithfulness scoring using answer–context overlap

Token-level attribution

Multi-query robustness checks

Automated abstention responses

RAG benchmarking dashboards

👩‍💻 Author
Sravani Bandela
Data Scientist | NLP & GenAI
GitHub: https://github.com/sravani2003


