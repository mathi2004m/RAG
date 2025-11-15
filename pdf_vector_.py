import faiss
import ollama
import numpy as np
import pickle
from pypdf import PdfReader


# ------------------------------
# 1. Extract PDF Text
# ------------------------------
def extract_pdf_text(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    all_text = ""
    page_lengths = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        all_text += text
        page_lengths.append(len(text))

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(all_text):,}")

    return all_text, total_pages, page_lengths


# ------------------------------
# 2. Chunk Text
# ------------------------------
def chunk_text(text, chunk_size=500):
    chunks = []
    metadata = []

    total_length = len(text)
    approx_pages = max(1, total_length // chunk_size)

    for i in range(0, total_length, chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

        # Rough page estimation
        estimated_page = (i // (total_length // approx_pages)) + 1

        metadata.append({
            "start_pos": i,
            "estimated_page": estimated_page
        })

    print(f"✂️ Created {len(chunks)} chunks")
    return chunks, metadata


# ------------------------------
# 3. Generate Embeddings via Ollama
# ------------------------------
def get_embedding(text):
    try:
        resp = ollama.embeddings(
            model="mistral",
            prompt=text
        )
        return np.array(resp["embedding"], dtype="float32")
    except Exception as e:
        print("❌ Embedding error:", e)
        return None


def build_embeddings(chunks):
    vectors = []

    print("🔄 Generating embeddings using Mistral (Ollama)...")

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}...")

        emb = get_embedding(chunk)

        if emb is None:
            print("⚠️ Skipping chunk due to embedding failure.")
            continue

        vectors.append(emb)

    vectors = np.array(vectors)
    print("🔢 Embedding matrix shape:", vectors.shape)

    return vectors


# ------------------------------
# 4. Build FAISS Index
# ------------------------------
def create_faiss_index(vectors):
    dim = vectors.shape[1]
    print(f"🗂️ Creating FAISS index with dimension = {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index


# ------------------------------
# 5. Save Everything
# ------------------------------
def save_database(index, chunks, metadata, total_pages):
    faiss.write_index(index, "vectors.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata,
            "total_pages": total_pages
        }, f)

    print("💾 Saved: vectors.index + chunks.pkl")


# ------------------------------
# 6. Master Function
# ------------------------------
def pdf_to_vectors(pdf_path):
    text, total_pages, page_lengths = extract_pdf_text(pdf_path)
    chunks, metadata = chunk_text(text)
    vectors = build_embeddings(chunks)
    index = create_faiss_index(vectors)
    save_database(index, chunks, metadata, total_pages)

    print("🎉 Vector DB ready!")
    return vectors, chunks


# ------------------------------
# 7. Run
# ------------------------------
if __name__ == "__main__":
    pdf_file = "Data Warehouse System Manager.pdf "  # Change here
    pdf_to_vectors(pdf_file)

