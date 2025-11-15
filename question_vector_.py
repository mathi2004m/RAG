import faiss
import ollama
import numpy as np
import pickle
import os


def get_embedding(text):
    """Generate embedding using Ollama (Mistral)."""
    try:
        resp = ollama.embeddings(
            model="mistral",   # uses local mistral model
            prompt=text
        )

        if "embedding" not in resp:
            print("❌ Embedding Error: No 'embedding' field in response")
            print("Response:", resp)
            return None

        return np.array(resp["embedding"], dtype="float32")

    except Exception as e:
        print("❌ Embedding Error:", e)
        return None


def chat_with_mistral(context, question, total_pages):
    """Generate answer using Mistral (Ollama)."""
    try:
        resp = ollama.chat(
            model="mistral",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are answering questions about a {total_pages}-page document. "
                        f"Always mention page numbers when possible."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer based on the context:"
                }
            ]
        )

        if "message" not in resp:
            print("❌ Chat Error: No 'message' field in Ollama response")
            print("Response:", resp)
            return None

        if "content" not in resp["message"]:
            print("❌ Chat Error: No 'content' field in message")
            print("Response:", resp)
            return None

        return resp["message"]["content"]

    except Exception as e:
        print("❌ Chat Error:", e)
        return None


def ask_question(question):
    # Check if vector files exist
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Error: Vector database not found!")
        print("🔧 Please run 'pdf_to_vectors.py' first to create the database.")
        return None

    try:
        # Load saved data
        index = faiss.read_index("vectors.index")

        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        # Get embedding
        query_vector = get_embedding(question)

        if query_vector is None:
            print("❌ Could not generate embedding for question.")
            return None

        query_vector = query_vector.reshape(1, -1)

        # Search FAISS
        scores, indices = index.search(query_vector.astype('float32'), 3)

        print(f"🔍 Found {len(indices[0])} relevant chunks:")

        # Build context
        context_parts = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= len(chunks):
                print(f"⚠️ Invalid chunk index returned: {idx}")
                continue

            page_num = metadata[idx]['estimated_page']
            print(f"   Chunk {i + 1}: Score {score:.3f} (≈Page {page_num})")

            context_parts.append(f"[Page {page_num}]: {chunks[idx]}")

        context = "\n\n".join(context_parts)

        # Generate answer using Mistral
        answer = chat_with_mistral(context, question, total_pages)
        return answer

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return None


def main():
    # Check if vector DB exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found!")
        print("🔧 Please run 'pdf_to_vectors.py' first to create the database.")
        print("📋 Steps:")
        print("   1. Run: python pdf_to_vectors.py")
        print("   2. Then run: python question_vector_.py")
        return

    # Load database info
    try:
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        total_pages = data['total_pages']

        print(f"✅ Database loaded: {len(chunks)} chunks from {total_pages} pages")
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return

    print("\n" + "=" * 60)
    print("🤖 RAG System Ready! Ask me questions about your PDF")
    print("💡 Type 'bye', 'quit', 'exit', or 'q' to exit")
    print("🔢 Type 'info' to see database statistics")
    print("=" * 60)

    while True:
        question = input("\n❓ Your question: ").strip()

        if question.lower() in ['bye', 'quit', 'exit', 'q']:
            print("👋 Goodbye! Thanks for using the RAG system!")
            break

        if question.lower() == 'info':
            print("📊 Database Info:")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Total chunks: {len(chunks)}")
            print("   • Vector dimensions: Auto-detected (Ollama)")
            print(f"   • Average chunks per page: {len(chunks) / total_pages:.1f}")
            print(f"   • Sample chunk: {chunks[0][:100]}...")
            continue

        if not question:
            print("⚠️ Please enter a question!")
            continue

        print("🔍 Searching and generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"\n🤖 Answer: {answer}")
        else:
            print("❌ Sorry, I couldn't generate an answer. Please try again.")


if __name__ == "__main__":
    main()
