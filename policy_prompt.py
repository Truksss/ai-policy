from langchain.prompts import PromptTemplate

policy_prompt = PromptTemplate(
    template=(
        "You are a legislator and an expert in drafting responsible AI policies for schools.\n\n"
        "Details:\n"
        "- School/Organization: {school}\n"
        "- Country: {country}\n"
        "- Level: {level}\n"
        "- Requirements: {requirements}\n"
        "- Scope: {scope}\n\n"
        "Context from similar policies (authoritative sources):\n"
        "{context}\n\n"
        "Task:\n"
        "1. Draft a comprehensive AI policy **grounded strictly in the provided context**. "
        "Every major policy statement must be supported by the retrieved context. "
        "Do not invent new rules or policies that are not in the context.\n\n"
        "2. If important requirements are missing from the context, acknowledge the gap explicitly "
        "and mark them under a section called 'Additional Recommendations'. These should be clearly "
        "separated from the context-grounded policy.\n\n"
        "3. Present the final output in **structured markdown** with clear sections\n"
        "Make the policy clear, practical, and written in professional, legislative style."
    ),
    input_variables=["school", "country", "level", "requirements", "scope", "context"]
)
