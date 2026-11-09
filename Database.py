
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


# 1. Load all PDFs
pdf_folder = Path("./Data")
documents = []

for pdf_file in pdf_folder.rglob("*.pdf"):
    print("Loading:", pdf_file)

    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

print("\nTotal pages:", len(documents))


# 2. Chunk all documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print("Total chunks:", len(texts))


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Verify metadata before indexing
print("\nSample chunk metadata:")
print(texts[0].metadata)


# 5. Create FAISS database, preserving metadata
vectorstore = FAISS.from_documents(
    documents=texts,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)


# 6. Save FAISS database
vectorstore.save_local("faiss_db")

print("\nFAISS database created and saved successfully!")


# 7. Verify saved database with a test query
results = vectorstore.similarity_search(
    "What is the Code of Civil Procedure, 1908?",
    k=3
)

for i, result in enumerate(results, start=1):
    print(f"\n========== RESULT {i} ==========")
    print("Source:", result.metadata.get("source"))
    print("Page:", result.metadata.get("page"))
    print(result.page_content[:200])
