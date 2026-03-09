import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Load documents
with open("docs.json") as f:
    documents = json.load(f)


# ---------------------------
# Document Chunking
# ---------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------------------
# Prepare Text Corpus
# ---------------------------
all_text = []

for doc in documents:
    chunks = chunk_text(doc["content"])
    all_text.extend(chunks)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(all_text)


# ---------------------------
# Generate Embedding
# ---------------------------
def get_embedding(text):
    return vectorizer.transform([text]).toarray()[0]


# ---------------------------
# Build Vector Store
# ---------------------------
vector_store = []

for doc in documents:

    chunks = chunk_text(doc["content"])

    for chunk in chunks:

        embedding = get_embedding(chunk)

        vector_store.append({
            "title": doc["title"],
            "content": chunk,
            "embedding": embedding
        })


# ---------------------------
# Similarity Search
# ---------------------------
def search(query, top_k=3):

    query_embedding = get_embedding(query)

    scores = []

    for item in vector_store:

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        scores.append((similarity, item))

    scores.sort(reverse=True)

    return scores[:top_k]


# ---------------------------
# Chat API (RAG Pipeline)
# ---------------------------
@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.json
        message = data["message"]

        results = search(message)

        context = "\n".join([r[1]["content"] for r in results])

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{message}

If the answer is not in the context, say you don't have enough information.
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        answer = response.text

        return jsonify({
            "reply": answer,
            "retrievedChunks": len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "reply": str(e)
        }), 500


# ---------------------------
# Frontend Route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)