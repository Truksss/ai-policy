from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import QueryRequest, QueryResponse, PolicyGenerationRequest, PolicyGenerationResponse
from policy_agent import generate_policy
from rag_query import query_rag
from metrics_api import router as metrics_router

app = FastAPI(title="AI Policy Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000", 
    "https://ai-policy.onrender.com", 
    "https://policy-jylh.vercel.app/"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    # Pass optional chat history through for follow-up awareness
    history_payload = None
    if request.history:
        history_payload = [m.dict() for m in request.history]
    answer = query_rag(request.question, history=history_payload)
    return QueryResponse(answer=answer)

@app.post("/generate-policy", response_model=PolicyGenerationResponse)
def generate_policy_endpoint(request: PolicyGenerationRequest):
    return generate_policy(request)

# Include metrics router
app.include_router(metrics_router)    