"""
TrustRAG: A Self-Verifying RAG System with Hallucination Detection
--------------------------------------------------------------------
Run with:  streamlit run app.py

Everything here uses free, open-source models - no API keys needed.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="TrustRAG", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 960px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }
    .trustrag-hero {
        background: linear-gradient(135deg, #2E86DE 0%, #1B4F8C 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1.2rem;
    }
    .trustrag-hero h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-size: 2.1rem;
    }
    .trustrag-hero p {
        color: #E6F0FA;
        margin-bottom: 0;
        font-size: 1.02rem;
    }
    .chip-row { margin-top: 0.9rem; }
    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .usecase-card {
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: #F8F9FB;
        border-left: 5px solid var(--accent, #2E86DE);
        height: 100%;
        margin-bottom: 0.9rem;
    }
    .usecase-card h4 { margin: 0 0 0.35rem 0; }
    .usecase-card p { margin: 0 0 0.5rem 0; color: #3b3f45; font-size: 0.92rem; }
    .usecase-card code {
        background: #eef1f5;
        padding: 0.15rem 0.4rem;
        border-radius: 6px;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Cached model / data loaders (run once, reused across interactions)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading answer-generation model...")
def load_generation_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, model


@st.cache_resource(show_spinner="Loading hallucination-checker model...")
def load_nli_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    return tok, model


@st.cache_data(show_spinner="Loading sample contracts (CUAD dataset)...")
def load_sample_contracts(n=15):
    from datasets import load_dataset
    ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    df = ds["train"].to_pandas()
    titles = df["title"].unique().tolist()[:n]
    contracts = {}
    for t in titles:
        row = df[df["title"] == t].iloc[0]
        contracts[t] = row["context"]
    return contracts


def pretty_title(raw_title: str) -> str:
    cleaned = raw_title.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.title()
    if len(cleaned) > 60:
        cleaned = cleaned[:57] + "..."
    return cleaned


def extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF, DOCX, or TXT file."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        elif name.endswith(".docx"):
            import docx
            document = docx.Document(uploaded_file)
            return "\n".join(p.text for p in document.paragraphs)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return ""


# ----------------------------------------------------------------------
# Core pipeline functions
# ----------------------------------------------------------------------

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@st.cache_data(show_spinner="Indexing document...")
def build_index(_embed_model, chunks):
    import faiss
    embeddings = _embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings


def generate_answer(gen_tokenizer, gen_model, question, context_text):
    prompt = f"""Answer the question based only on the context below.

Context:
{context_text}

Question: {question}

Answer:"""
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=100)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)


def check_support(nli_tokenizer, nli_model, context, answer):
    inputs = nli_tokenizer(context, answer, return_tensors="pt", truncation=True, max_length=512)
    outputs = nli_model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    labels = ["contradiction", "neutral", "entailment"]
    predicted = labels[probs.argmax().item()]
    confidence = probs.max().item()
    return predicted, confidence


def run_pipeline(question, chunks, index, embed_model, gen_tokenizer, gen_model,
                  nli_tokenizer, nli_model, k=3):
    query_embedding = embed_model.encode([question])
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    context_text = "\n".join(retrieved_chunks)
    answer = generate_answer(gen_tokenizer, gen_model, question, context_text)

    chunk_verdicts = []
    best_verdict = "unsupported"
    best_confidence = 0.0
    best_chunk = None
    for c in retrieved_chunks:
        verdict, confidence = check_support(nli_tokenizer, nli_model, c, answer)
        chunk_verdicts.append((c, verdict, confidence))
        if verdict == "entailment" and confidence > best_confidence:
            best_verdict = "supported"
            best_confidence = confidence
            best_chunk = c

    return {
        "answer": answer,
        "verdict": best_verdict,
        "confidence": best_confidence,
        "supporting_chunk": best_chunk,
        "retrieved_chunks": chunk_verdicts,
    }


# ----------------------------------------------------------------------
# Real-world use cases shown in the gallery tab
# ----------------------------------------------------------------------

USE_CASES = [
    {
        "icon": "📜", "title": "Legal Contract Review", "color": "#2E86DE",
        "desc": "Find and verify clauses like termination, liability, or renewal terms without reading the whole contract.",
        "example": "What are the termination conditions?",
    },
    {
        "icon": "🏥", "title": "Insurance Policy Analysis", "color": "#16A085",
        "desc": "Check what's covered, excluded, or capped in a policy before filing a claim.",
        "example": "What is covered under this policy?",
    },
    {
        "icon": "🏦", "title": "Loan & Credit Agreements", "color": "#E67E22",
        "desc": "Verify interest rates, repayment schedules, and penalty clauses before signing.",
        "example": "What happens if a payment is missed?",
    },
    {
        "icon": "🧑‍💼", "title": "HR Policies & Handbooks", "color": "#8E44AD",
        "desc": "Answer employee questions about leave, benefits, or conduct policies accurately.",
        "example": "How many sick leave days are provided?",
    },
    {
        "icon": "🏠", "title": "Rental & Lease Agreements", "color": "#C0392B",
        "desc": "Confirm deposit rules, maintenance responsibilities, and notice periods.",
        "example": "What is the notice period for ending this lease?",
    },
    {
        "icon": "🩺", "title": "Medical Consent Forms", "color": "#2C3E50",
        "desc": "Make sure patients or staff can verify exactly what a form states before signing.",
        "example": "What risks are disclosed in this consent form?",
    },
]


# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="trustrag-hero">
        <h1>🛡️ TrustRAG</h1>
        <p>Upload any document and ask questions — every answer is checked against the
        source text before it's shown to you, so you know when to trust it.</p>
        <div class="chip-row">
            <span class="chip">100% Free</span>
            <span class="chip">No API Keys</span>
            <span class="chip">3 AI Models Working Together</span>
            <span class="chip">Built-in Hallucination Check</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar: document setup
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("1. Choose a document")
    source = st.radio(
        "Document source",
        ["Sample legal contract", "Upload your own file", "Paste your own text"],
    )

    doc_text = None
    if source == "Sample legal contract":
        contracts = load_sample_contracts()
        chosen_title = st.selectbox(
            "Pick a contract", list(contracts.keys()), format_func=pretty_title
        )
        doc_text = contracts[chosen_title]

    elif source == "Upload your own file":
        uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        if uploaded is not None:
            with st.spinner("Extracting text..."):
                doc_text = extract_text_from_upload(uploaded)
            if doc_text:
                st.caption(f"Extracted {len(doc_text):,} characters from {uploaded.name}")
            else:
                st.warning("No readable text found in this file.")

    else:
        doc_text = st.text_area("Paste document text here", height=220)

    st.divider()
    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider("Chunk size (characters)", 200, 1000, 500, step=100)
        overlap = st.slider("Chunk overlap", 0, 150, 50, step=10)
        top_k = st.slider("Chunks to retrieve (k)", 1, 5, 3)

# ----------------------------------------------------------------------
# Main tabs
# ----------------------------------------------------------------------

tab_analyze, tab_usecases, tab_stress, tab_about = st.tabs(
    ["📄 Analyze Document", "💡 Real-World Use Cases", "🧪 Stress Test", "ℹ️ How It Works"]
)

question = ""  # defined early so it's accessible in the stress-test tab too

with tab_analyze:
    if not doc_text:
        st.info("Choose, upload, or paste a document in the sidebar to get started.")
    else:
        embed_model = load_embed_model()
        gen_tokenizer, gen_model = load_generation_model()
        nli_tokenizer, nli_model = load_nli_model()

        chunks = split_into_chunks(doc_text, chunk_size, overlap)
        index, _ = build_index(embed_model, chunks)

        m1, m2, m3 = st.columns(3)
        m1.metric("Chunks indexed", len(chunks))
        m2.metric("Characters analyzed", f"{len(doc_text):,}")
        m3.metric("Chunks checked per answer", top_k)

        st.write("")
        with st.container(border=True):
            st.subheader("Ask a question about this document")
            question = st.text_input(
                "Your question", placeholder="e.g. Who are the parties to this agreement?"
            )
            run = st.button("Get Answer", type="primary")

            if run and question:
                with st.spinner("Retrieving, generating, and verifying..."):
                    result = run_pipeline(
                        question, chunks, index, embed_model,
                        gen_tokenizer, gen_model, nli_tokenizer, nli_model, k=top_k,
                    )

                st.markdown("**Answer**")
                st.write(result["answer"])

                if result["verdict"] == "supported":
                    st.success(f"✅ SUPPORTED — confidence {result['confidence']:.0%}")
                    if result["supporting_chunk"]:
                        with st.expander("Show the source text that supports this answer"):
                            st.write(result["supporting_chunk"])
                else:
                    st.error(
                        "❌ UNSUPPORTED — no retrieved chunk clearly confirms this answer. "
                        "Treat this answer with caution; it may be a hallucination."
                    )

                with st.expander("Show all retrieved chunks and their individual verdicts"):
                    for i, (chunk_text, verdict, conf) in enumerate(result["retrieved_chunks"]):
                        badge = {"entailment": "🟢", "neutral": "🟡", "contradiction": "🔴"}[verdict]
                        st.markdown(f"**Chunk {i+1}** {badge} `{verdict}` (confidence: {conf:.0%})")
                        st.caption(chunk_text[:400] + ("..." if len(chunk_text) > 400 else ""))
                        st.divider()

with tab_usecases:
    st.subheader("Where this kind of system is actually useful")
    st.caption(
        "Upload a document from any of these categories and ask the example question — "
        "or your own — to see it in action."
    )
    cols = st.columns(2)
    for i, uc in enumerate(USE_CASES):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="usecase-card" style="--accent: {uc['color']};">
                    <h4>{uc['icon']} {uc['title']}</h4>
                    <p>{uc['desc']}</p>
                    <code>Try: "{uc['example']}"</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_stress:
    st.subheader("🧪 Does it catch a wrong answer?")
    st.caption("Enter a deliberately false statement and confirm the checker correctly rejects it.")

    if not doc_text:
        st.info("Load a document in the sidebar first.")
    else:
        fake_claim = st.text_input(
            "A false claim to test", placeholder="e.g. This contract was signed by Microsoft in 2050."
        )
        test_btn = st.button("Test this claim")

        if test_btn and fake_claim:
            embed_model = load_embed_model()
            nli_tokenizer, nli_model = load_nli_model()
            chunks = split_into_chunks(doc_text, chunk_size, overlap)
            index, _ = build_index(embed_model, chunks)

            with st.spinner("Checking..."):
                query_embedding = embed_model.encode([question or fake_claim])
                distances, indices = index.search(np.array(query_embedding), top_k)
                caught = False
                rows = []
                for idx in indices[0]:
                    c = chunks[idx]
                    verdict, confidence = check_support(nli_tokenizer, nli_model, c, fake_claim)
                    rows.append(
                        {"chunk": c[:200] + "...", "verdict": verdict, "confidence": f"{confidence:.0%}"}
                    )
                    if verdict == "entailment":
                        caught = True

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if caught:
                st.warning("⚠️ This claim was marked as supported by at least one chunk — review it.")
            else:
                st.success("✅ Correctly rejected — no chunk supports this false claim.")

with tab_about:
    st.subheader("How TrustRAG works")
    steps = [
        ("1. Chunk", "The document is split into small overlapping pieces, like index cards."),
        ("2. Embed", "Each piece is converted into numbers representing its meaning."),
        ("3. Retrieve", "Your question is matched against those pieces to find the most relevant ones."),
        ("4. Generate", "An AI model reads the relevant pieces and writes a plain-English answer."),
        ("5. Verify", "A separate AI model checks the answer against each source piece, one at a time."),
        ("6. Verdict", "You see ✅ Supported or ❌ Unsupported — never just a confident-sounding guess."),
    ]
    for title, desc in steps:
        c1, c2 = st.columns([1, 5])
        c1.markdown(f"**{title}**")
        c2.write(desc)
    st.divider()
    st.caption(
        "Models used: all-MiniLM-L6-v2 (embeddings), google/flan-t5-base (answer generation), "
        "roberta-large-mnli (hallucination verification). All free and open-source via HuggingFace."
    )
