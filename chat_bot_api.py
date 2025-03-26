from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import faiss
import numpy as np
import hashlib
import uuid
import shutil
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str

class AssessmentResponse(BaseModel):
    assessment: str
    condition: str

class SummaryResponse(BaseModel):
    issue: str
    confidence: str

os.environ["GROQ_API_KEY"] = "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"
GROQ_MODEL = "llama3-70b-8192"

sessions = {}

PDF_DIR = "uploaded_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

app = FastAPI(title="Healthcare Chatbot API", 
              description="API for healthcare conversations with PDF support",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_session(session_id=None):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        dimension = 100
        index = faiss.IndexFlatL2(dimension)
        
        sessions[session_id] = {
            "id": session_id,
            "messages": [],
            "pdf_store": {
                "index": index,
                "documents": [],
                "metadata": []
            },
            "has_pdf": False
        }
    
    return sessions[session_id], session_id

def text_to_vector(text, dimension=100):
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    
    vector = np.zeros(dimension)
    for i in range(min(dimension, len(hash_hex))):
        vector[i] = int(hash_hex[i], 16) / 16.0
    
    return vector

def process_pdf(file_path, session):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        for i, doc in enumerate(docs):
            embedding = text_to_vector(doc.page_content)
            
            session["pdf_store"]["index"].add(np.array([embedding]))
            session["pdf_store"]["documents"].append(doc)
            session["pdf_store"]["metadata"].append({
                "page": i+1,
                "total_pages": len(docs)
            })
        
        session["has_pdf"] = True
        return len(docs)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 0

def search_pdf(query, session, top_k=3):
    if not session["has_pdf"]:
        return "No PDF has been uploaded yet."
    
    query_vector = text_to_vector(query)
    D, I = session["pdf_store"]["index"].search(np.array([query_vector]), top_k)
    
    results = []
    for idx in I[0]:
        if idx < len(session["pdf_store"]["documents"]):
            doc = session["pdf_store"]["documents"][idx]
            metadata = session["pdf_store"]["metadata"][idx]
            page_info = f"[Page {metadata['page']}/{metadata['total_pages']}]"
            results.append(f"{page_info} {doc.page_content}")
    
    return "\n\n".join(results)

def generate_assessment(messages):
    user_inputs = [msg["content"] for msg in messages if msg["role"] == "user"]
    all_content = "\n".join(user_inputs)
    
    llm = ChatGroq(model=GROQ_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a healthcare assistant providing a potential diagnostic assessment.
        Based on the conversation, identify the most likely condition that matches the symptoms.
        Format your response as:
        
        ### Based on your symptoms, you may have:
        
        **[Condition name]**
        
        **Key symptoms identified**:
        - [symptom 1]
        - [symptom 2]
        - [symptom 3]
        
        **IMPORTANT**: This is not a medical diagnosis. Please consult a healthcare professional.
        """),
        ("human", "{conversation}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    assessment = chain.invoke({"conversation": all_content})
    
    condition = "Medical condition"
    for line in assessment.split("\n"):
        if line.strip() and not line.startswith("**") and not line.startswith("#") and not line.startswith("-"):
            condition = line.strip()
            break
    
    return assessment, condition

def create_chat_chain(system_message, pdf_context=None):
    if pdf_context:
        system_message = f"{system_message}\n\nUse the following PDF context:\n{pdf_context}"
    
    llm = ChatGroq(model=GROQ_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{message}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

@app.get("/")
async def root():
    return {"message": "Healthcare Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session, session_id = get_session(request.session_id)
    
    session["messages"].append({"role": "user", "content": request.message})
    
    if session["has_pdf"] and any(kw in request.message.lower() for kw in ["pdf", "document", "file", "read", "what does it say"]):
        pdf_context = search_pdf(request.message, session)
        system_message = "You are a helpful healthcare assistant analyzing a medical document."
        chain = create_chat_chain(system_message, pdf_context)
    else:
        system_message = "You are a helpful healthcare assistant. Provide clear medical information but always advise consulting a doctor for specific concerns."
        chain = create_chat_chain(system_message)
    
    chat_history = []
    for msg in session["messages"][:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    
    try:
        response = chain.invoke({
            "chat_history": chat_history,
            "message": request.message
        })
        
        session["messages"].append({"role": "assistant", "content": response})
        
        return {"message": response, "session_id": session_id}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    session, session_id = get_session(session_id)
    
    try:
        file_path = os.path.join(PDF_DIR, f"{session_id}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        page_count = process_pdf(file_path, session)
        
        welcome_message = f"📄 I've processed your PDF: {file.filename} ({page_count} pages). You can now ask me questions about this document!"
        
        session["messages"].append({"role": "assistant", "content": welcome_message})
        
        return {
            "message": welcome_message,
            "session_id": session_id,
            "page_count": page_count
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.post("/assessment")
async def get_assessment(session_id: str):
    session, _ = get_session(session_id)
    
    if len(session["messages"]) < 2:
        return HTTPException(status_code=400, detail="Not enough conversation history for assessment")
    
    try:
        assessment, condition = generate_assessment(session["messages"])
        
        assessment_message = (
            "# 🏥 YOUR FINAL ASSESSMENT\n\n"
            f"{assessment}\n\n"
            "---\n\n"
            f"### Copy this condition to find a doctor: `{condition}`\n\n"
            "1. Copy the condition above\n"
            "2. Paste the condition in the search box to find a specialist"
        )
        
        session["messages"].append({"role": "assistant", "content": assessment_message})
        
        return {
            "assessment": assessment,
            "condition": condition,
            "message": assessment_message
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error generating assessment: {str(e)}")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    session, _ = get_session(session_id)
    return {"messages": session["messages"]}

@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    session, _ = get_session(session_id)
    
    pdf_store = session["pdf_store"]
    has_pdf = session["has_pdf"]
    
    session["messages"] = []
    session["pdf_store"] = pdf_store
    session["has_pdf"] = has_pdf
    
    return {"message": "Session reset successfully"}

@app.post("/summary/{session_id}", response_model=SummaryResponse)
async def get_summary(session_id: str):
    session, _ = get_session(session_id)
    
    if len(session["messages"]) < 2:
        return HTTPException(status_code=400, detail="Not enough conversation history for summary")
    
    try:
        user_inputs = [msg["content"] for msg in session["messages"] if msg["role"] == "user"]
        all_content = "\n".join(user_inputs)
        
        llm = ChatGroq(model=GROQ_MODEL)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a healthcare assistant. 
            Analyze the conversation and identify the most likely medical issue or condition in ONE SHORT SENTENCE.
            Also provide a confidence level (high, medium, or low) based on the clarity of symptoms.
            Format your response exactly like this:
            
            {"issue": "Patient likely has [condition]", "confidence": "[high/medium/low]"}
            
            Be concise and direct. Do not include explanations or disclaimers in the response.
            """),
            ("human", "{conversation}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"conversation": all_content})
        
        import json
        try:
            parsed = json.loads(result)
            issue = parsed.get("issue", "Unable to determine specific medical issue")
            confidence = parsed.get("confidence", "low")
        except Exception as json_error:
            print(f"JSON parsing error: {json_error}, Response was: {result}")
            issue = "Unable to determine specific medical issue"
            confidence = "low"
        
        return SummaryResponse(issue=issue, confidence=confidence)
    except Exception as e:
        return SummaryResponse(
            issue="Error analyzing conversation", 
            confidence="low"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)