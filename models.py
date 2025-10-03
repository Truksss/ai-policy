from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None

class QueryResponse(BaseModel):
    answer: str

class PolicyUploadRequest(BaseModel):
    school: str
    country: str
    level: str
    source: Optional[str] = None 

class PolicyUploadResponse(BaseModel):
    message: str
    policy_id: Optional[str] = None

class PolicyGenerationRequest(BaseModel):
    school: str
    country: str
    level: str
    requirements: Optional[str] = None  # e.g., "focus on student privacy"
    scope: Optional[List[str]] = None  # e.g., ["teachers", "students"]

class PolicyGenerationResponse(BaseModel):
    generated_policy: str  # markdown text
    metadata: Dict