import os
import pickle
import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
from huggingface_hub import hf_hub_download
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
APP_PASSWORD   = os.environ.get("APP_PASSWORD", "engineer123")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
DATASET_REPO   = "sachingomber/engineering-books-index"
PORT           = int(os.environ.get("PORT", 7860))

bm25           = None
metadata       = None
chunk_offsets  = None   # numpy array of byte offsets - tiny in RAM
chunks_file    = None   # open file handle - stays on disk
gemini_model   = None
index_ready    = False

SYSTEM_PROMPT = """You are a warm and precise engineering reference assistant with access to 283 engineering textbooks covering thermodynamics, boilers, steam generation, heat exchangers, refrigeration, food processing, water treatment, and fluid mechanics.

STRICT RULES:
1. Answer ONLY using facts present in the SOURCE PASSAGES provided.
2. If the answer is not found say: I do not find that topic in your engineering library.
3. NEVER add facts from your own training knowledge.
4. Speak naturally and conversationally, 2 to 4 sentences max.
5. Be warm and clear. Avoid robotic tone.
6. Never say passage 1 or source 2. Integrate facts naturally."""


def download_index():
    global bm25, metadata, chunk_offsets, chunks_file, index_ready
    log.info("Downloading index files from Hugging Face...")
    # hf_hub_download with local_dir="/tmp" + filename="bm25/file.pkl"
    # saves file to /tmp/bm25/file.pkl  ← correct path
    local_dir = "/tmp"
    bm25_dir  = "/tmp/bm25"
    os.makedirs(bm25_dir, exist_ok=True)

    files_needed = [
        "bm25/bm25_index.pkl",
        "bm25/metadata.pkl",
        "bm25/chunks.jsonl",
        "bm25/chunks_offsets.npy",
    ]

    for hf_path in files_needed:
        local_path = f"{local_dir}/{hf_path}"
        if not os.path.exists(local_path):
            log.info(f"Downloading {hf_path}...")
            hf_hub_download(
                repo_id=DATASET_REPO, filename=hf_path,
                repo_type="dataset", token=HF_TOKEN,
                local_dir=local_dir
            )
            log.info(f"Downloaded to {local_path}")

    log.info("Loading BM25 index into RAM (~305MB)...")
    with open(f"{bm25_dir}/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    log.info("Loading metadata into RAM...")
    with open(f"{bm25_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    log.info("Loading chunk offsets into RAM (tiny)...")
    chunk_offsets = np.load(f"{bm25_dir}/chunks_offsets.npy")

    log.info("Opening chunks.jsonl file handle (stays on disk)...")
    chunks_file = open(f"{bm25_dir}/chunks.jsonl", "r", encoding="utf-8")

    log.info(f"Index ready! {len(metadata):,} chunks available.")
    index_ready = True


def get_chunk_text(i: int) -> str:
    """Read a single chunk from disk by index - O(1), minimal RAM."""
    chunks_file.seek(int(chunk_offsets[i]))
    return json.loads(chunks_file.readline())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        log.info("Gemini configured.")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, download_index)
    yield
    if chunks_file:
        chunks_file.close()


app = FastAPI(title="Engineering Voice Assistant", lifespan=lifespan)


def search_books(query: str, n: int = 5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    results = []
    for i in top_idx:
        if scores[i] > 0:
            text = get_chunk_text(i)
            results.append({
                "text": text,
                "book_name": metadata[i]["book_name"],
                "category":  metadata[i]["category"],
                "paragraph_num": metadata[i]["paragraph_num"],
                "relevance": round(float(scores[i]), 2)
            })
    return results


def build_prompt(query: str, passages: list) -> str:
    context = "\n\n---\n\n".join([
        f"[{p['category']} / {p['book_name']} - Para {p['paragraph_num']}]\n{p['text'][:1200]}"
        for p in passages
    ])
    return f"{SYSTEM_PROMPT}\n\nSOURCE PASSAGES:\n{context}\n\nUSER QUESTION: {query}\n\nANSWER:"


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "index_ready": index_ready,
                         "chunks": len(metadata) if metadata else 0})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
    if token != APP_PASSWORD:
        await websocket.close(code=4001, reason="Invalid password")
        return
    await websocket.accept()
    log.info("Client connected")
    try:
        while True:
            raw  = await websocket.receive_text()
            msg  = json.loads(raw)

            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg.get("type") == "query":
                query = msg.get("text", "").strip()
                if not query:
                    continue

                if not index_ready:
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Library loading, please wait 1-2 min..."}))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Searching 283 engineering books..."}))

                try:
                    passages = search_books(query)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                    continue

                if not passages:
                    await websocket.send_text(json.dumps({
                        "type": "answer",
                        "text": "I could not find relevant information for that query. Try using specific engineering terms.",
                        "citations": [], "query": query}))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "status", "message": "Generating answer..."}))

                try:
                    response = gemini_model.generate_content(
                        build_prompt(query, passages),
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3, max_output_tokens=400)
                    )
                    answer = response.text.strip()
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
                    passages = []

                await websocket.send_text(json.dumps({
                    "type": "answer", "text": answer,
                    "citations": [
                        {"book_name": p["book_name"], "category": p["category"],
                         "paragraph_num": p["paragraph_num"], "relevance": p["relevance"]}
                        for p in passages[:3]
                    ],
                    "query": query
                }))

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    p = Path("static/index.html")
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>Starting up...</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)