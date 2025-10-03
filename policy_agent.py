from rag_index import build_or_load_index
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from models import PolicyGenerationRequest, PolicyGenerationResponse
from policy_prompt import policy_prompt
from metrics import RAGASMetricsCollector
import threading

db = build_or_load_index()
retriever = db.as_retriever(search_kwargs={"k": 20})
llm = ChatOpenAI(model="gpt-4o-mini")

def generate_policy(request: PolicyGenerationRequest) -> PolicyGenerationResponse:
    query = f"AI policy for {request.level} schools in {request.country}"
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content[:1200] for d in retrieved_docs])

    chain = LLMChain(llm=llm, prompt=policy_prompt)
    result = chain.run(
        school=request.school,
        country=request.country,
        level=request.level,
        requirements=request.requirements or "None specified",
        scope=", ".join(request.scope or []),
        context=context,
    )

    # Async metrics collection
    def _collect_policy_metrics_async(generated_result, retrieved_docs_list, country):
        try:
            metrics_collector = RAGASMetricsCollector()
            policy_metrics = metrics_collector.collect_policy_metrics(
                generated_result,
                [{"content": d.page_content, **d.metadata} for d in retrieved_docs_list],
                country
            )
            metrics_collector.save_metrics(policy_metrics, "policy")
        except Exception as e:
            print(f"Error collecting policy metrics asynchronously: {e}")

    thread = threading.Thread(
        target=_collect_policy_metrics_async,
        args=(result, retrieved_docs, request.country),
    )
    thread.daemon = True
    thread.start()

    return PolicyGenerationResponse(
        generated_policy=result,
        metadata={
            "school": request.school,
            "country": request.country,
            "level": request.level,
            "requirements": request.requirements,
            "retrieved_sources": [d.metadata for d in retrieved_docs],
        }
    )
