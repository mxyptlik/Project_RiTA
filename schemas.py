from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    sender: str
    response: str
    session_id: str
    sources: Optional[List[str]] = None
    type: str

class StreamEvent(BaseModel):
    content: Optional[str] = None
    complete: Optional[bool] = None
    error: Optional[str] = None