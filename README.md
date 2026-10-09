# TrustRAG: A Self-Verifying Retrieval-Augmented QA System with Hallucination Detection

TrustRAG retrieves relevant chunks from a document, generates an answer, then checks
whether that answer is actually **supported** by the retrieved text using an NLI
(entailment) model. Answers with no supporting evidence are flagged as possible hallucinations.

100% free and open-source. No API keys, no paid services.

## Files in this package

| File | What it is |
|---|---|
| `TrustRAG_Evaluation.ipynb` | Colab notebook: builds the pipeline step by step and measures the hallucination rate on real CUAD contracts (the project's results) |
| `TrustRAG_App_Colab.ipynb` | Colab notebook: launches the interactive web app with a free public link |
| `app.py` | The Streamlit web app (upload a document, ask questions, see verified answers) |
| `requirements.txt` | Python dependencies for running the app locally |
| `.streamlit/config.toml` | Blue colour theme for the app |
| `PROJECT_SHEET.md` | Ready-to-submit project details |

## How it works
1. **Chunk**: the document is split into small overlapping pieces.
2. **Embed**: each chunk is converted to a vector using `all-MiniLM-L6-v2`.
3. **Retrieve**: the question is embedded and the closest chunks are found with FAISS.
4. **Generate**: `google/flan-t5-base` writes an answer from the retrieved chunks.
5. **Verify**: `roberta-large-mnli` checks the answer against each chunk individually.
6. **Verdict**: Supported if at least one chunk entails the answer, otherwise Unsupported.

Checking chunks individually matters: checking all chunks combined into one block
produced false "contradiction" results during testing.

## Option A: Run in Google Colab (recommended)
1. Go to https://colab.research.google.com and choose File > Upload notebook.
2. Upload `TrustRAG_App_Colab.ipynb`.
3. Click Runtime > Run all, or run the cells top to bottom in order.
4. In the Step 6 output, click the `trycloudflare.com` link to open the app.
   Keep the notebook running while you use the app.

To reproduce the evaluation results instead, upload `TrustRAG_Evaluation.ipynb`
and run it the same way.

## Option B: Run locally
Requires Python 3.9+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The first run downloads about 2.5 GB of models.

## Using the app
- **Analyze Document**: choose a sample contract, upload a PDF/DOCX/TXT, or paste text, then ask a question.
- **Real-World Use Cases**: examples of where this applies (legal, insurance, loans, HR, leases, medical forms).
- **Stress Test**: enter a false statement and confirm the checker rejects it.
- **How It Works**: the pipeline explained step by step.

## Known limitations
- Scanned or image-only PDFs have no extractable text (OCR is not included).
- `flan-t5-base` is a small model. Long or oddly phrased questions can produce non-answers, which the verifier correctly marks as Unsupported.
- The NLI model reads about 512 tokens at a time, so each chunk is checked separately.

## Dataset
CUAD (Contract Understanding Atticus Dataset), The Atticus Project, CC BY 4.0.
https://huggingface.co/datasets/theatticusproject/cuad-qa
