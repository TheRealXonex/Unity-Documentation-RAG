from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

# =========================
# LOAD JSONL
# =========================

df = pd.read_json("unity_docs.jsonl", lines=True)

# =========================
# EMBEDDING MODEL
# =========================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# DATABASE LOCATION
# =========================

db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

# =========================
# CREATE VECTOR STORE
# =========================

vector_store = Chroma(
    collection_name="unity_docs",
    persist_directory=db_location,
    embedding_function=embeddings
)

# =========================
# ADD DOCUMENTS
# =========================

if add_documents:

    documents = []
    ids = []

    for i, row in df.iterrows():

        document = Document(

            page_content=f"""
            Title: {row['title']}

            Section: {row['section']}

            Content:
            {row['content']}
            """,

            metadata={
                "title": row["title"],
                "section": row["section"],
                "url": row["url"],
                "path": row["path"],
                "version": row["version"],
                "doc_type": row["doc_type"]
            }
        )

        documents.append(document)
        ids.append(str(i))

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print("Documents added.")

else:

    print("Database already exists.")
    
# =========================
# TEST SEARCH
# =========================

class UnityDocsRetriever:
    def __init__(self):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vector_store = Chroma(
            collection_name="unity_docs",
            persist_directory="./chroma_langchain_db",
            embedding_function=embeddings
        )
        
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
    
    def search(self, query: str):
        return self.retriever.invoke(query)