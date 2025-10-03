from rag_index import build_or_load_index
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List, Optional
from metrics import RAGASMetricsCollector
import threading
import time

db = build_or_load_index()
retriever = db.as_retriever(search_kwargs={"k": 20})

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template=(
        "You are an assistant helping summarize AI policies for schools worldwide.\n"
        "Use the following context from policy documents to answer the question.\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Instructions:\n"
        "- If the context directly answers, provide a clear, concise answer.\n"
        "- If the answer is not explicit, summarize the closest relevant information.\n"
        "- Always mention which schools/countries the policies came from.\n"
        "- Prefer content whose country or school matches the query terms.\n"
        "- Don't just say 'I don't know'. If unsure, explain what is available and missing, and cite the nearest relevant country/school found.\n"
    ),
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

def rewrite_question_with_history(question: str, history: Optional[List[dict]]) -> str:
    if not history:
        return question
    # Simple heuristic rewriter: append prior QA context to bias retrieval
    recent_turns = []
    for msg in history[-6:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            recent_turns.append(f"{role}: {content}")
    history_hint = " | ".join(recent_turns)
    return f"[Follow-up based on: {history_hint}] {question}"


def collect_metrics_async(query: str, answer: str, retrieved_docs: List):
    """Collect metrics in a separate thread to avoid blocking the response"""
    def run_metrics():
        try:
            print(f"Starting async metrics collection for query: {query[:50]}...")
            metrics_collector = RAGASMetricsCollector()
            rag_metrics = metrics_collector.collect_rag_metrics(query, answer, retrieved_docs)
            metrics_collector.save_metrics(rag_metrics, "rag")
            print(f"Completed async metrics collection for query: {query[:50]}")
        except Exception as e:
            print(f"Error in async metrics collection: {e}")
    
    # Start metrics collection in background thread
    thread = threading.Thread(target=run_metrics)
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()

def query_rag(query: str, history: Optional[List[dict]] = None):
    expanded_query = rewrite_question_with_history(query, history)
    result = qa_chain.invoke({"query": expanded_query})
    answer = result["result"]
    retrieved_docs = result.get("source_documents", [])
    
    # Start metrics collection asynchronously (non-blocking)
    collect_metrics_async(query, answer, retrieved_docs)
    
    return answer