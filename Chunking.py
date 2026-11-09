from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_folder = Path("./Data")

documents = []

for pdf_file in pdf_folder.rglob("*.pdf"):
    print("Loading:", pdf_file)

    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

print("Total pages loaded:", len(documents[0].page_content))
print("Total pages loaded:", (documents[0].page_content))




text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print("Total pages:", len(documents))
print("Total chunks:", len(texts))