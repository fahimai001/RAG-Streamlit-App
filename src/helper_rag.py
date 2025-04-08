import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def load_environment_variables():
    load_dotenv()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY")
    }

def initialize_llm_and_embeddings(api_key):
    llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    return llm, embeddings

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store_and_retriever(docs, embeddings, client=None):
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    return retriever

def define_rag_chain(retriever, llm):
    template = (
        "You are an assistant for summarizing and extracting key information from documents.\n"
        "Use the following pieces of retrieved context to identify the most important sections relevant to the question.\n"
        "If you cannot identify the key sections, just say that you don't know.\n"
        "Document Context: {context}\n"
        "Question: {question}\n"
        "Task: Extract the most important chunks from the document relevant to the question.\n"
        "Chunks:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain