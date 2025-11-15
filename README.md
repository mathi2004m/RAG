# RAG
This repository contains a complete PDF Question-Answering RAG system powered by Mistral (Ollama). It extracts text from PDFs, chunks and embeds the content, stores embeddings using FAISS, and uses an LLM to generate context-aware answers. A simple, fast, offline RAG implementation ideal for learning and local experimentation.
How It Works:
1. You give a PDF.
   We extract all the text and break it into small readable pieces (chunks).
2. We embed each chunk.
   Ollama’s Mistral model converts every chunk into a vector — basically a numerical representation of the meaning.
3. We store everything in FAISS.
   FAISS quickly finds which chunks are most similar to your question.
4. You ask a question.
   Your question is also converted into a vector and compared with all stored chunk vectors.
5. We fetch the top 3 relevant chunks.
   These chunks provide real context to the LLM.
6. Mistral generates the final answer.
   It reads the retrieved context and answers in a grounded, meaningful way.

This project demonstrates how to build a simple, working RAG system using:
    Ollama Mistral for embeddings + LLM
    FAISS for vector search
    Python for PDF processing and retrieval logic
