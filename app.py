import os
import re
import pandas as pd
import gradio as gr
from fastapi import FastAPI, Request
from gradio.routes import mount_gradio_app
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- File paths ---
SALES_XLSX = "sales.xlsx"
FORM_TXT = "sales_analysis_formulas.txt"
FORM_SUM_CSV = "sales_analysis_formulas_summary.csv"
FORM_EX_CSV = "sales_analysis_formulas_with_examples.csv"

# --- Load dataset ---
if os.path.exists(SALES_XLSX):
    sales_df = pd.read_excel(SALES_XLSX)
else:
    sales_df = None

# --- Load formula text files ---
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

form_txt = read_file(FORM_TXT)
form_sum = read_file(FORM_SUM_CSV)
form_ex = read_file(FORM_EX_CSV)

# --- Chunk text into manageable pieces for embeddings ---
def chunk_text(text, chunk_size=800):
    text = text.strip()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] if text else []

docs = []
for i, chunk in enumerate(chunk_text(form_txt)):
    docs.append({"id": f"txt_{i}", "text": chunk, "meta": {"source": "formulas"}})
for i, chunk in enumerate(chunk_text(form_sum)):
    docs.append({"id": f"sum_{i}", "text": chunk, "meta": {"source": "summary"}})
for i, chunk in enumerate(chunk_text(form_ex)):
    docs.append({"id": f"ex_{i}", "text": chunk, "meta": {"source": "examples"}})

# --- Create small sales summaries ---
if sales_df is not None and "Package" in sales_df.columns:
    pkg_agg = sales_df.groupby("Package").agg({"GrossAfterCb": "sum", "Profit": "sum"}).reset_index().head(200)
    for _, row in pkg_agg.iterrows():
        docs.append({
            "id": f"pkg_{row['Package']}",
            "text": f"Package {row['Package']} — Revenue {row['GrossAfterCb']}, Profit {row['Profit']}",
            "meta": {"source": "sales"}
        })

# --- Setup Chroma (in-memory, free-tier friendly) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("sales_knowledge")
collection.upsert(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
    metadatas=[d["meta"] for d in docs]
)

# --- Load small model (good for Render) ---
MODEL = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# --- RAG Answer ---
def rag_answer(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    retrieved_docs = results["documents"][0] if results and "documents" in results else []
    context = "\n".join(retrieved_docs)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer clearly and show formulas if needed."
    output = generator(prompt, max_new_tokens=200, do_sample=False)
    return output[0]["generated_text"]

# --- Deterministic logic ---
def top_packages_by_profit(n=10):
    if sales_df is None:
        return "Sales file not loaded."
    df = sales_df.dropna(subset=["Package", "Profit"])
    top = df.groupby("Package")["Profit"].sum().reset_index().sort_values("Profit", ascending=False).head(n)
    return top.to_string(index=False)

# --- Unified query handler ---
def respond(query):
    q = query.lower()
    if "top" in q and "package" in q:
        m = re.search(r"top\s*(\d+)", q)
        n = int(m.group(1)) if m else 10
        return top_packages_by_profit(n)
    return rag_answer(query)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Sales Analyst Assistant")
    inp = gr.Textbox(label="Ask a question", placeholder="Example: Top 10 packages by profit")
    btn = gr.Button("Ask")
    out = gr.Textbox(label="Answer")
    btn.click(respond, inputs=inp, outputs=out)

# --- FastAPI + Gradio Integration (for ChatGPT API calls) ---
app = FastAPI()

@app.post("/run")
async def run_query(request: Request):
    data = await request.json()
    query = data.get("query", "")
    answer = respond(query)
    return {"response": answer}

# Mount the Gradio UI
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


