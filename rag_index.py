import os
import re
import fitz
import json
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

INDEX_DIR = "data/ai_policies_index"

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    return text.strip()

def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = " ".join([page.get_text("text") for page in doc])
    return clean_text(text)

def extract_webpage(url: str) -> str:
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return clean_text(soup.get_text())

def build_or_load_index(base_folder: str = "data/policies", force_rebuild: bool = False):
    embeddings = OpenAIEmbeddings()

    # Allow forcing a rebuild via function arg or env var
    env_force = os.getenv("FORCE_REBUILD", "0").lower() in {"1", "true", "yes"}
    force = force_rebuild or env_force

    if os.path.exists(INDEX_DIR) and not force:
        print("Loading existing FAISS index")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("Building new FAISS index")
    policies = []

    for level in os.listdir(base_folder):
        level_path = os.path.join(base_folder, level)
        if not os.path.isdir(level_path):
            continue

        for country in os.listdir(level_path):
            country_path = os.path.join(level_path, country)
            if not os.path.isdir(country_path):
                continue

            for file in os.listdir(country_path):
                if not file.endswith(".pdf"):
                    continue

                path = os.path.join(country_path, file)
                school = os.path.splitext(file)[0]
                content = extract_pdf(path)

                policies.append({
                    "content": content,
                    "metadata": {
                        "school": school,
                        "country": country,
                        "level": level,
                        "source": path
                    }
                })

    with open("data/web.json", "r") as f:
        webpages = json.load(f)

    for w in webpages:
        content = extract_webpage(w["url"])
        policies.append({
            "content": content,
            "metadata": {
                "school": w["school"],
                "country": w["country"],
                "level": w["level"],
                "source": w["url"]
            }
        })

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = []
    for p in policies:
        meta = p["metadata"]
        header = (
            f"Level: {meta['level']}\n"
            f"Country: {meta['country']}\n"
            f"School: {meta['school']}\n"
        )
        for i, chunk in enumerate(splitter.split_text(p["content"])):
            docs.append(Document(
                page_content=header + "\n" + chunk,
                metadata={**meta, "chunk": i}
            ))

    print(f"Created {len(docs)} chunks")

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_DIR)
    return db