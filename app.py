import os
import sqlite3
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
APP_PASSWORD   = os.environ.get("APP_PASSWORD", "engineer123")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
DATASET_REPO   = "sachingomber/engineering-books-index"
PORT           = int(os.environ.get("PORT", 7860))
DB_PATH        = "/tmp/sqlite/books.db"

db_conn      = None
gemini_model = None
index_ready  = False

SYSTEM_PROMPT = """You are a warm and precise engineering reference assistant with access to 283 engineering textbooks covering thermodynamics, boilers, steam generation, heat exchangers, refrigeration, food processing, water treatment, and fluid mechanics.

STRICT RULES:
1. Answer ONLY using facts present in the SOURCE PASSAGES provided.
2. If the answer is not found say: I do not find that topic in your engineering library.
3. NEVER add facts from your own training knowledge.
4. Speak naturally and conversationally, 2 to 4 sentences max.
5. Be warm and clear. Avoid robotic tone.
6. Never say passage 1 or source 2. Integrate facts naturally."""


def download_index():
    global db_conn, index_ready
    log.info("Downloading SQLite database from Hugging Face...")
    try:
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename="sqlite/books.db",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp"
        )
        log.info("Download complete. Opening database...")
        db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row
        # Test query
        count = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        log.info(f"Index ready! {count:,} chunks in SQLite FTS5.")
        index_ready = True
    except Exception as e:
        log.error(f"Failed to load index: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-3.6-flash")
        log.info("Gemini configured.")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, download_index)
    yield
    if db_conn:
        db_conn.close()


app = FastAPI(title="Engineering Voice Assistant", lifespan=lifespan)


def search_books(query: str, n: int = 5):
    # Escape FTS5 special chars
    safe_query = query.replace('"', '""')
    rows = db_conn.execute(
        """SELECT text, book_name, category, paragraph_num,
                  rank as relevance
           FROM books
           WHERE books MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (safe_query, n)
    ).fetchall()
    return [
        {"text": r["text"], "book_name": r["book_name"],
         "category": r["category"], "paragraph_num": r["paragraph_num"],
         "relevance": round(abs(float(r["relevance"])), 2)}
        for r in rows
    ]


def build_prompt(query: str, passages: list) -> str:
    context = "\n\n---\n\n".join([
        f"[{p['category']} / {p['book_name']} - Para {p['paragraph_num']}]\n{p['text'][:1200]}"
        for p in passages
    ])
    return f"{SYSTEM_PROMPT}\n\nSOURCE PASSAGES:\n{context}\n\nUSER QUESTION: {query}\n\nANSWER:"


@app.get("/health")
async def health():
    count = 0
    if db_conn:
        try:
            count = db_conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        except Exception:
            pass
    return JSONResponse({"status": "ok", "index_ready": index_ready, "chunks": count})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
    if token != APP_PASSWORD:
        await websocket.close(code=4001, reason="Invalid password")
        return
    await websocket.accept()
    log.info("Client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

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
                    # FTS5 syntax error - retry with simple words
                    try:
                        simple = " ".join(query.split()[:5])
                        passages = search_books(simple)
                    except Exception:
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