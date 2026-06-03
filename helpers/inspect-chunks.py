import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Path to your saved DB (same as ingestion)
PERSIST_DIR = "dbv2/chroma_db"


def load_vectorstore():
    """Load existing Chroma DB (NO COST)"""
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )

    return vectorstore


def inspect_all_chunks():
    """Print all chunks with IDs and text"""

    db = load_vectorstore()

    # get everything stored
    data = db.get(include=["documents", "metadatas"])

    for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):

        print("\n" + "=" * 80)
        print(f"CHUNK INDEX: {i}")
        print(f"CHUNK ID: {meta.get('chunk_id')}")
        print("=" * 80)

        # your summarised chunk content
        print(doc[:1000])


def inspect_specific_chunk(chunk_id: int):
    """Find and print a specific chunk"""

    db = load_vectorstore()

    data = db.get(include=["documents", "metadatas"])

    for doc, meta in zip(data["documents"], data["metadatas"]):

        if meta.get("chunk_id") == chunk_id:

            print("\n" + "=" * 80)
            print(f"FOUND CHUNK ID: {chunk_id}")
            print("=" * 80)

            print(doc)
            return

    print(f"Chunk {chunk_id} not found")


def search_chunks(query: str, k: int = 5):
    """Cheap semantic search over stored chunks"""

    db = load_vectorstore()

    results = db.similarity_search(query, k=k)

    for i, doc in enumerate(results):

        print("\n" + "=" * 80)
        print(f"RESULT {i+1}")
        print("=" * 80)

        print(doc.page_content[:1000])

        print("\nMETADATA:")
        print(doc.metadata)


if __name__ == "__main__":

    print("Choose mode:")
    print("1. Inspect all chunks")
    print("2. Inspect specific chunk")
    print("3. Search chunks")

    choice = input("> ")

    if choice == "1":
        inspect_all_chunks()

    elif choice == "2":
        cid = int(input("Chunk ID: "))
        inspect_specific_chunk(cid)

    elif choice == "3":
        q = input("Query: ")
        search_chunks(q)