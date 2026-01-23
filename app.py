import streamlit as st
from ingest import ingest_pdf, build_chunks
from retriever import build_vector_store, retrieve
from evaluator import evaluate_retrieval
from llm import call_llm as call_grok

st.sidebar.header("⚙️ Guardrails")
threshold = st.sidebar.slider(
    "Grounding threshold (min retrieval confidence to allow answer)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01
)
st.sidebar.caption("Below threshold → abstain to reduce hallucination risk.")

st.title("🔍 RAG Quality Inspector (FAISS + TF-IDF + Groq)")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    pages = ingest_pdf("temp.pdf")
    chunks = build_chunks(pages, chunk_size=800, overlap=150)
    store = build_vector_store(chunks)

    query = st.text_input("Ask a question")

    if query:
        results = retrieve(query, store, k=4)
        eval_result = evaluate_retrieval(results)
        confidence = eval_result.get("confidence", 0.0)
        
        passed = confidence >= threshold

        st.subheader("Retrieval Confidence")
        st.metric("Confidence", f"{confidence:.2f}")

        if passed:
            st.success(f"✅ Grounded enough (>= {threshold:.2f}) → generation allowed")
        else:
            st.error(f"⚠️ Low grounding (< {threshold:.2f}) → abstaining to avoid hallucinations")


        if not eval_result["pass"]:
            st.error("⚠️ Abstained: insufficient grounding (low retrieval confidence)")
        else:
            context = "\n\n".join([chunk for chunk, _ in results])
            prompt = f"""
Answer the question using ONLY the context below.
If the answer is not grounded in the context, say you don't know.

Context:
{context}

Question:
{query}
"""
            answer = call_grok(prompt)
            st.subheader("Answer")
            st.write(answer)

        # st.subheader("Retrieved Chunks")
        # for chunk, score in results:
        #     st.write(chunk[:350])
        #     st.caption(f"Cosine score: {round(score, 2)}")
        st.subheader("Retrieved Evidence")
        for i, (chunk, score) in enumerate(results, start=1):
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown(f"### #{i}")
                    st.metric("Similarity", f"{score:.2f}")
                with c2:
                    preview = chunk.strip().replace("\n", " ")
                    preview = preview[:450] + ("..." if len(preview) > 450 else "")
                    st.write(preview)

                    with st.expander("View full chunk"):
                        st.text(chunk)

