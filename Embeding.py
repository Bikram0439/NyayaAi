from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# 1. Load all PDFs
# ==========================================

pdf_folder = Path("./Data")

documents = []

for pdf_file in pdf_folder.rglob("*.pdf"):
    print("Loading:", pdf_file)

    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

print("\nTotal pages:", len(documents))


# ==========================================
# 2. Chunk all documents
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print("Total chunks:", len(texts))


# ==========================================
# 3. Create embedding model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 4. Create embeddings
# ==========================================

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in texts]
)

print("Total vectors:", len(vectors))

print("\nFirst vector:")
print(vectors[0])