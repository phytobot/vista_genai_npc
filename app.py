from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with open("lore.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)
app = Flask(__name__)
CORS(app)
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_input = data['message']
    query_embedding = embedding_model.encode(user_input, convert_to_tensor=True).cpu().numpy()
    _, top_indices = index.search(np.array([query_embedding]), k=2)
    relevant_chunks = "\n".join([documents[i] for i in top_indices[0]])
    prompt = (
        f"You are a wise, magical NPC in a fantasy game. A player approaches and says:\n"
        f"\"{user_input}\"\n\n"
        f"Use your ancient knowledge below to respond:\n{relevant_chunks}\n\n"
        f"Reply with mystical advice, staying in character:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    npc_reply = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()
    return jsonify({
        'retrieved_lore': relevant_chunks,
        'npc_reply': npc_reply
    })

if __name__ == '__main__':
    app.run(debug=True)
