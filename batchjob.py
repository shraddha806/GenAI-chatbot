import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


def build_vector_index():
    # Step --> 1: Load the PDF file
    # Ensure the path to the PDF file is correct
    feil_path = os.path.join("documents", "tcs.pdf")
    loader = PyPDFLoader(feil_path)
    documents = loader.load()

    print(len(documents))

    # Step --> 2: Convert the documents to Chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)

    print("total Chunks :", len(chunks))

    # Step --> 3: Create embeddings for the chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Created embeddings for the chunks...")

    # Step --> 4: create Faiss Database
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Step 5 --> store chunks in the Faiss database
    vector_store.add_documents(chunks)
    print("Chunks added to the vector store...")

    # Step 6: Store the Vector db permanently
    vector_store.save_local("tcs_doc_index")
    print("Vector store saved successfully.")


if __name__ == "__main__":
    build_vector_index()