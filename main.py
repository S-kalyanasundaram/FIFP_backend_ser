import re
import os
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.schema import Document
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi import Body
from pydantic import BaseModel
from typing import List, Dict, Optional

# -----------------------------
# Config / ENV
# -----------------------------
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not mongo_uri or not OPENAI_API_KEY:
    raise RuntimeError("❌ MONGO_URI or OPENAI_API_KEY missing in .env")

# MongoDB Client
client = MongoClient(mongo_uri)
DB_NAME = "FIRE"
COLLECTIONS = [
    "networths", "personalrisks", "net_worths", "multiusers", "mfdetails",
    "marriagefundplans", "insurances", "houseplans", "googles", "fundallocations",
    "childexpenses", "childeducations", "budgetincomeplans", "firequestions",
    "financials", "customplans", "expensesmasters", "vehicles", "emergencyfunds",
    "profiles", "realitybudgetincomes"
]

# Chat History Collection
CHAT_HISTORY_COLLECTION = "chat_history"

# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI App



# FastAPI App
app = FastAPI(title="Finance QA API", version="1.1")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# In-memory cache for retrievers (per user)
retriever_cache: Dict[str, Dict] = {}

# -----------------------------
# Helper Functions
# -----------------------------
def validate_user_id(user_id: str):
    """Validate user_id is a non-empty string."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(status_code=400, detail="Invalid user_id format.")

def cosine_sim(a: np.ndarray, b_matrix: np.ndarray):
    """Compute cosine similarity between a vector and a matrix."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b_matrix / (np.linalg.norm(b_matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)

def embed_texts_openai(texts: List[str], model: str = "text-embedding-3-small"):
    """Generate embeddings using OpenAI."""
    try:
        resp = openai_client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")

def generate_answer_openai(context: str, question: str, model: str = "gpt-4o-mini"):
    """Generate answer using OpenAI Chat model."""
    system_prompt = (
        "You are a helpful personal finance assistant. "
        "ONLY use the provided CONTEXT (from the user's database). "
        "If the answer is not present, reply exactly with NO_ANSWER. "
        "Do not invent facts. Mask sensitive info (phones, cards, emails)."
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer:"

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI completion error: {e}")

def mask_sensitive(text: str):
    """Mask emails, phone numbers, and card numbers."""
    text = re.sub(r'([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', r'\1***\2', text)
    text = re.sub(r'(\+?\d[\d\-\s]{8,}\d)', lambda m: "****" + m.group(0)[-4:], text)
    text = re.sub(r'((?:\d[ -]?){13,19})', lambda m: "**** **** **** " + re.sub(r'\D', '', m.group(0))[-4:], text)
    return text

def format_answer(question: str, answer: str) -> str:
    """
    Format the AI answer into a ChatGPT-style structured markdown response.
    Detects key-value pairs and makes them user friendly.
    """

    # Clean extra spaces
    answer = answer.strip()

    # Try to pretty-format dictionary-like text into markdown
    def format_key_values(text: str) -> str:
        # Matches things like: key: value
        lines = text.split()
        if "{" in text and "}" in text:
            # Handle raw dict-style responses
            text = text.replace("{", "").replace("}", "")
        parts = re.split(r"(\w+:\s*[^,]+)", text)
        formatted_lines = []
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                formatted_lines.append(f"- **{k.strip()}**: {v.strip()}")
        return "\n".join(formatted_lines) if formatted_lines else text

    # Beautify the answer if it contains key:value structure
    if ":" in answer:
        formatted_answer = format_key_values(answer)
    else:
        formatted_answer = answer

    # Final wrapper in ChatGPT style
    formatted = f"""
## 💡 Answer to Your Question  

**Question Asked:**  
➡️ {question}  

**📌 Here’s what I found:**  
{formatted_answer}  

---

✨ *Tip:* Stay consistent with your savings & review your plan regularly! 
"""
    return formatted



def load_user_documents(user_id: str):
    db = client[DB_NAME]
    all_docs = []
    for collection_name in COLLECTIONS:
        try:
            cursor = db[collection_name].find({"userId": user_id}, projection={"_id": 0, "userId": 0})
            for doc in cursor:
                text = f"[{collection_name}]\n" + "\n".join([f"{k}: {v}" for k, v in doc.items()])
                all_docs.append(Document(page_content=text))
        except Exception as e:
            print(f"Error loading {collection_name}: {e}")
    return all_docs

def build_local_retriever(docs: List[Document]):
    """Build local in-memory retriever for a user."""
    if not docs:
        return {"docs": [], "embeddings": np.array([])}
    texts = [d.page_content for d in docs]
    vectors = embed_texts_openai(texts)
    return {"docs": docs, "embeddings": vectors}

def answer_from_db_local(
    retriever_obj: Dict,
    question: str,
    top_k: int = 3,
    sim_threshold: float = 0.05   # lowered from 0.12
):
    """Find answer from local retriever."""
    doc_embeddings = retriever_obj.get("embeddings", np.array([]))
    docs = retriever_obj.get("docs", [])
    if doc_embeddings.size == 0 or not docs:
        return None, "❌ No data found for this user."

    # Normalize question (simple intent mapping)
    normalized_q = question.lower().strip()
    synonyms = {
        "monthly income": "my monthly income",
        "salary": "my monthly income",
        "income": "my monthly income",
        "earnings": "my monthly income"
    }
    if normalized_q in synonyms:
        question = synonyms[normalized_q]

    qvec = embed_texts_openai([question])[0]
    sims = cosine_sim(qvec, doc_embeddings)
    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    top_texts = [docs[i].page_content for i in top_idx]

    if len(top_scores) == 0 or float(np.max(top_scores)) < sim_threshold:
        return None, "❌ Could not find relevant data."

    context = "\n\n".join(top_texts)[:8000]
    raw = generate_answer_openai(context=context, question=question)

    if "NO_ANSWER" in raw:
        return None, "❌ I couldn’t find that in your data."

    answer = mask_sensitive(raw)
    answer = (answer.replace("$", "₹")
              .replace("USD", "INR")
              .replace("usd", "INR")
              .replace("dollars", "rupees")
              .replace("Dollars", "Rupees"))

    if answer.isdigit() or answer.replace("₹", "").strip().isdigit():
        answer = f"✅ Your monthly income is {answer} INR. Great to know your earnings!"

    return answer, None



def save_chat_history(user_id: str, question: str, answer: str, session_id: Optional[str] = None):
    """Save chat history in MongoDB under a session."""
    db = client[DB_NAME]
    if not session_id:
        session_id = str(ObjectId())
    chat_entry = {
        "userId": user_id,
        "sessionId": session_id,
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    }
    db[CHAT_HISTORY_COLLECTION].insert_one(chat_entry)
    return session_id

# -----------------------------
# API Models
# -----------------------------
class QuestionRequest(BaseModel):
    user_id: str
    question: str
    session_id: Optional[str] = None

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Finance QA API is running"}

@app.post("/load-data/{user_id}")
def load_data(user_id: str):
    validate_user_id(user_id)
    docs = load_user_documents(user_id)
    if not docs:
        raise HTTPException(status_code=404, detail="No documents found for this user")
    retriever_obj = build_local_retriever(docs)
    retriever_cache[user_id] = retriever_obj
    return {"status": "success", "message": f"Data loaded for user {user_id}", "doc_count": len(docs)}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_id = req.user_id.strip()
    question = req.question.strip()
    session_id = req.session_id

    # Check greetings / default questions
    greetings = ["hi", "hello", "hey", "hii", "hiii", "hola"]
    greetings2 = ["ok", "bye", "bya", "goodbay", "thankyou", "tq","good"]
    
    if question.lower() in greetings:
        answer = f"{question} 👋! I can help you only with your finance-related queries."
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}
    if question.lower() in greetings2:
        answer = f"{question} Great! Let's keep your money working for you. 💹"
        session_id = save_chat_history(user_id, question, answer, session_id)
        return {"answer": answer, "session_id": session_id}

    # Load retriever if not already cached
    if user_id not in retriever_cache:
        docs = load_user_documents(user_id)
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found for this user")
        retriever_cache[user_id] = build_local_retriever(docs)

    retriever_obj = retriever_cache[user_id]
    answer, err = answer_from_db_local(retriever_obj, question)

    # ❌ No fallback – only finance-related dataset answers
    if not answer:
        answer = "❌ I can only answer finance-related questions based on your data."

    # Save history
    session_id = save_chat_history(user_id, question, answer, session_id)
    clean_answer = re.sub(r'\s+', ' ', answer).strip()
    formatted_answer = format_answer(question, clean_answer)
    return {"answer": formatted_answer, "session_id": session_id}



@app.get("/chat-sessions/{user_id}")
def get_sessions(user_id: str):
    db = client[DB_NAME]
    sessions = (
        db[CHAT_HISTORY_COLLECTION]
        .aggregate([
            {"$match": {"userId": user_id}},  # ✅ match as string
            {"$group": {
                "_id": "$sessionId",
                "latest": {"$max": "$timestamp"},
                "title": {"$first": "$question"}
            }},
            {"$sort": {"latest": -1}}
        ])
    )
    result = []
    for s in sessions:
        title = s.get("title") or "Untitled"
        result.append({
            "session_id": s["_id"],
            "title": title[:50] + ("..." if len(title) > 50 else ""),
            "latest": s["latest"].isoformat() if s.get("latest") else None
        })
    return {"sessions": result}

@app.get("/chat-history/{user_id}/{session_id}")
@app.get("/chat-history/{user_id}/{session_id}")
def chat_history(user_id: str, session_id: str):
    db = client[DB_NAME]
    chats = (
        db[CHAT_HISTORY_COLLECTION]
        .find({"userId": user_id, "sessionId": session_id})  # ✅ match as string
        .sort("timestamp", 1)
    )
    history = []
    for c in chats:
        c["_id"] = str(c["_id"])
        c["timestamp"] = c["timestamp"].isoformat() if c.get("timestamp") else None
        history.append(c)
    return {"history": history}




@app.delete("/chat-sessions/{user_id}/{session_id}")
def delete_session(user_id: str, session_id: str):
    db = client[DB_NAME]
    result = db[CHAT_HISTORY_COLLECTION].delete_many(
        {"userId": user_id, "sessionId": session_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session deleted"}


@app.put("/chat-sessions/{user_id}/{session_id}")
def rename_session(user_id: str, session_id: str, data: dict = Body(...)):
    new_title = data.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")

    db = client[DB_NAME]
    first_msg = db[CHAT_HISTORY_COLLECTION].find_one(
        {"userId": user_id, "sessionId": session_id},
        sort=[("timestamp", 1)]
    )
    if not first_msg:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update the first message question as title
    db[CHAT_HISTORY_COLLECTION].update_one(
        {"_id": first_msg["_id"]},
        {"$set": {"question": new_title}}
    )
    return {"status": "success", "message": "Session renamed"}



