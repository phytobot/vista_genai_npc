from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from elevenlabs import save, set_api_key, generate
import torch
import faiss
import numpy as np
import os
import uuid
import json

# ElevenLabs API setup
set_api_key("sk_7b9ceafe714cadaa748742d10706d800ce7bb0f78d26ea2b")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load language model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load lore or context
with open("lore.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()

# FAISS index
faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
faiss_index.add(doc_embeddings)

# Flask setup
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

SESSION_FILE = 'session_memory.json'

# Serve index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve JS, assets, etc.
@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

# Chat logic for Trevor
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_input = data['message']

    # Retrieve relevant lore
    query_embedding = embedding_model.encode(user_input, convert_to_tensor=True).cpu().numpy()
    _, top_indices = faiss_index.search(np.array([query_embedding]), k=2)

    relevant_chunks = "\n".join([documents[i] for i in top_indices[0]])

    # Prompt for generation
    prompt = (
        f"You are Trevor Philips from Grand Theft Auto V. "
        f"Speak in a chaotic, violent, and sarcastic tone. "
        f"You're talking to a stranger who just said: \"{user_input}\"\n\n"
        f"Here’s some background info:\n{relevant_chunks}\n\n"
        f"Trevor:"
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    npc_reply = full_output.split("Trevor:")[-1].strip()

    # Generate TTS using ElevenLabs
    audio_filename = f"audio_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join("audio", audio_filename)
    os.makedirs("audio", exist_ok=True)

    audio = generate(
        text=npc_reply,
        voice="bIHbv24MWmeRgasZH58o",
        model="eleven_monolingual_v1"
    )
    save(audio, audio_path)

    return jsonify({
        'retrieved_lore': relevant_chunks,
        'npc_reply': npc_reply,
        'audio_url': f"/audio/{audio_filename}"
    })

@app.route('/audio/<filename>')
def get_audio(filename):
    audio_path = os.path.join("audio", filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg")
    else:
        return jsonify({'error': 'Audio not found'}), 404

if __name__ == '__main__':
    port = 5000
    print(f"Serving Phaser game at http://localhost:{port}/")
    app.run(debug=True, port=port)
