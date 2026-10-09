# Project Sheet

**Project Title:** TrustRAG: A Self-Verifying Retrieval-Augmented QA System with Hallucination Detection

**Problem Statement:** Standard RAG systems retrieve relevant document chunks and generate answers, but never verify that the generated answer is actually supported by the retrieved content. The LLM can hallucinate details, misquote figures, or merge facts incorrectly even when the correct source material was retrieved. This is risky in high-stakes domains like legal, insurance, or medical document Q&A, where a confident but wrong answer is worse than no answer at all.

**Objectives / Expected Outcome:** Build a RAG pipeline that retrieves document chunks and generates answers, then adds a verification layer using an NLI (entailment) model to check whether each answer is supported by the retrieved text. Classify answers as Supported / Unsupported and report a measurable hallucination rate across test queries, along with how retrieval settings (chunk size, top-k) affect it. Deliver an interactive web app that also lets users upload their own documents.

**Dataset Name:** CUAD (Contract Understanding Atticus Dataset), legal contract Q&A

**Dataset Link:** https://huggingface.co/datasets/theatticusproject/cuad-qa

**Dataset Source / License:** The Atticus Project, CC BY 4.0

**Proposed Method / Tools / Libraries:** Document chunking, Sentence-Transformers embeddings (all-MiniLM-L6-v2), FAISS vector retrieval, answer generation with Flan-T5-base, hallucination check with RoBERTa-large-MNLI (per-chunk entailment). Python, HuggingFace Transformers, FAISS, PyPDF2, python-docx, Streamlit.

**GitHub / Repository Link:** (add your repository link here)

**Status:** Proposed
