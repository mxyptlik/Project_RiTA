import os
import logging
import asyncio
from dotenv import load_dotenv
import json
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import APIRouter
from typing import Optional, Union

# Langchain Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# Utility Imports
from rapidfuzz import process, fuzz 
from schemas import ChatRequest, ChatResponse
from redis_manager import RedisSessionManager  
import time
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Local Imports
from chroma_service import chroma_service

load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry tracing
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# Disable anonymous telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- Environment Variable Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(',')

# Load system prompt
try:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
        logger.info(f"Loaded system prompt from {SYSTEM_PROMPT_PATH}")
except FileNotFoundError:
    logger.warning(f"{SYSTEM_PROMPT_PATH} not found; using default prompt.")
    SYSTEM_PROMPT = (
        "You are RiTA, Rising Tide Africa AI assistant. Use provided context to answer queries accurately and conversationally. "
        "If context doesn't contain the answer, say so. Always be friendly and include a relevant call-to-action."
    )
except Exception as e:
    logger.error(f"Error loading system prompt: {e}", exc_info=True)
    SYSTEM_PROMPT = "Error: Could not load system prompt."

# Redis session manager
redis_manager = RedisSessionManager()

# --- FAQ Section ---
FAQS = {
    "What is Rising Tide Africa?":
        "Rising Tide Africa is a women-centric angel investment network that educates, mentors, and invests in female-founded or gender-diverse startups across Africa. Would you like to learn more about our mission?",
    "Who founded Rising Tide Africa?":
        "Rising Tide Africa was co-founded by leading women angel investors, including Yemi Keri and Ndidi Nnoli‑Edozien, aiming to increase women’s participation in angel investing. Interested in their backgrounds?",
    "Where is Rising Tide Africa headquartered?":
        "Our headquarters are located in Victoria Island, Lagos, Nigeria. Would you like directions or office hours?",
    "What is the mission of Rising Tide Africa?":
        "Our mission is to build a New Africa by empowering women investors to support early-stage startups with capital, education, mentorship, and networking. Want details on a specific pillar?",
    "Who can join Rising Tide Africa?":
        "Membership is open to accredited women investors, mentors, and entrepreneurs endorsed by existing members. Want to explore membership tiers?",
    "How do I apply to become a member?":
            "You can apply via introduction from a current member or an online submission, both subject to endorsement by an existing member. Need the application link?",
    "What types of startups does RTA invest in?":
        "We focus on early-stage, digitally-enabled African startups that are female-founded, led, or gender-diverse. Curious about our portfolio?",
    "How much does RTA typically invest in each startup?":
        "Our investments are angel-level, varying per startup. We usually co-invest through our managed investment pool. Would you like a range?",
    "What kind of support do portfolio companies receive?":
        "Startups gain capital, strategic guidance, mentorship, governance support, network access, and visibility at RTA events. Interested in any specific benefit?",
    "What are the main pillars of RTA?":
        "Our four key pillars are Mentoring, Investment, Networking, and Education (e.g., Entrepreneurship 101 Accelerator). Want details on any?",
    "What is Deal Day?":
        "Deal Day is a bi-annual pitch event where investment-ready entrepreneurs present to angel investors for possible funding and mentoring. Want the upcoming dates?",
    "What are Masterclasses?":
        "RTA Masterclasses are intensive training sessions for startup owners on business fundamentals and capital readiness. Want next session topics?",
    "What are Mentorship Clinics?":
        "Mentorship Clinics pair entrepreneurs with experienced investors to receive real-world business advice and feedback. Want to sign up?",
    "What educational programs does RTA offer?":
        "Besides Masterclasses, we offer Entrepreneurship 101 and investor training to empower women as sophisticated angels. Want the curriculum?",
    "How is Rising Tide Africa structured?":
        "RTA is a managed investment pool run by a committee and overseen by a Board of Directors and Executive Committee. Interested in member roles?",
    "How many members are in the network?":
        "We consist of a growing community of women investors across Africa, Europe, North America, and Latin America—numbering in the hundreds. Want regional breakdown?",
    "How can I connect with Rising Tide Africa events?":
        "You can attend Deal Days, networking events, open houses and Masterclasses—both virtual and in-person. Interested in upcoming events?",
    "Can men participate in any RTA activities?":
        "Our core focus is empowering women. However, we collaborate with partners and mentors who support our mission—sometimes reaching beyond gender. Want partner info?",
    "Who are RTA’s partners?":
        "We partner with organizations like Ingressive, VC4A, SBH, ACEN, cchub, FCMB, Sterling Bank and more. Would you like an introduction?",
    "Can I mentor through Rising Tide Africa?":
        "Yes—especially during SheVentures and Mentorship Clinics. We often call for mentors via social channels and newsletter. Want the next call?",
    "How does RTA measure its impact?":
        "We track metrics like capital deployed, startups supported, female investor growth, and enterprise outcomes. Want to see our latest report?",
    "Where can I see RTA’s portfolio companies?":
        "You can find companies like Afrikrea, Bankly, Amayi Foods, Aruwa, Betty, Big Cabal Media and more on our website. Want links?",
    "How do I apply for accelerator or investment opportunities?":
        "Applications for Deal Day or accelerator are posted on our website and channels. We review startups quarterly. Shall I send deadlines?",
    "How can I stay updated with RTA news?":
        "Follow us on LinkedIn, Instagram, Medium, and subscribe to our newsletter via risingtideafrica.com. Want the links?",
    "How can I contact Rising Tide Africa?":
        "You can email info@risingtideafrica.com or use the contact form on our website. Need specific team contacts?"
}

def get_faq_answer(query: str) -> Optional[str]:
    match = process.extractOne(
        query.lower(), [q.lower() for q in FAQS.keys()], scorer=fuzz.token_sort_ratio, score_cutoff=90
    )
    if not match:
        return None
    for orig, ans in FAQS.items():
        if orig.lower() == match[0]:
            return ans
    return None

# --- Web Search Tool ---
web_search_tool = DuckDuckGoSearchRun()

# --- RAG Chain Components ---
def format_docs(docs):
    if not docs:
        return "No context available."
    return "\n\n".join(
        f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\nContent: {doc.page_content}"
        for doc in docs
    )

def get_retriever():
    retriever = chroma_service._retriever
    if retriever is None:
        raise RuntimeError("ChromaDB retriever is not initialized. Make sure the service is running correctly.")
    return retriever

# --- Application Setup ---
app = FastAPI(
    title="RiTA - Rising Tide Africa AI Assistant",
    description="API for interacting with RiTA, the AI assistant.",
    version="1.0.0"
)

# Add Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Instrument FastAPI for OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# --- Langchain RAG Chain ---
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.4, streaming=True)

# Main RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# RAG chain that passes documents through for source citation.
# It's structured to preserve the input 'question' and 'history' at each step.
rag_chain = (
    RunnablePassthrough.assign(
        documents=lambda x: get_retriever().invoke(x["question"])
    )
    .assign(context=lambda x: format_docs(x["documents"]))
    .assign(output=rag_prompt | llm | StrOutputParser())
)

# Chain with message history management
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    redis_manager.get_session_history_sync,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="output",
)

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        
        # 1. Check for FAQ match
        faq_answer = get_faq_answer(request.message)
        if faq_answer:
            # Since we are bypassing the main chain, we must handle history manually.
            await redis_manager.add_user_message(session_id, request.message)
            await redis_manager.add_ai_message(session_id, faq_answer)
            
            async def faq_stream():
                """Streams a FAQ answer in the expected NDJSON format."""
                yield json.dumps({"type": "sources", "data": ["FAQ"]}) + "\n"
                yield json.dumps({"type": "content", "data": faq_answer}) + "\n"
            
            return StreamingResponse(faq_stream(), media_type="application/x-ndjson")

        # 2. Execute RAG chain for streaming response
        config = {"configurable": {"session_id": session_id}}
        
        response_generator = chain_with_history.astream(
            {"question": request.message},
            config=config
        )
        
        async def stream_response():
            """Streams RAG chain output, sending sources first, then content."""
            sources_sent = False
            async for chunk in response_generator:
                # Yield sources once if they exist in the chunk
                if not sources_sent and "documents" in chunk:
                    documents = chunk.get("documents", [])
                    if documents: # Ensure there are documents before processing
                        seen_sources = set()
                        for doc in documents:
                            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                            if source_file not in seen_sources:
                                seen_sources.add(source_file)
                        
                        sources = list(seen_sources)
                        if sources:
                            yield json.dumps({"type": "sources", "data": sources}) + "\n"
                    sources_sent = True

                # Yield content chunk if it exists
                if "output" in chunk and chunk["output"]:
                    yield json.dumps({"type": "content", "data": chunk["output"]}) + "\n"

        return StreamingResponse(stream_response(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to RiTA, the Rising Tide Africa AI Assistant API."}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/favicon.ico")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes a user's session history from Redis.
    """
    try:
        await redis_manager.delete_history(session_id)
        logger.info(f"Successfully deleted session: {session_id}")
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session.")

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # Initialize services
    chroma_service.initialize()
    logger.info("ChromaDB service initialized.")
    await redis_manager.get_client()  # Initialize Redis connection pool
    logger.info("Redis connection pool initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    await redis_manager.close()  # Correctly await the async close method
    logger.info("Redis connection closed.")
    
    # Shut down the OpenTelemetry tracer provider to prevent event loop errors
    tracer_provider.shutdown()
    logger.info("OpenTelemetry tracer provider shut down.")

if __name__ == "__main__":
    import uvicorn
    # Note: The host is set to 0.0.0.0 to be accessible within a Docker container
    uvicorn.run(app, host="0.0.0.0", port=8001)
