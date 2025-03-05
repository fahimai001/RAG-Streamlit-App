import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def load_environment_variables():
    load_dotenv()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "weaviate_api_key": os.getenv("WEAVIATE_API_KEY"),
        "weaviate_url": os.getenv("WEAVIATE_URL")
    }

def initialize_llm_and_embeddings(api_key):
    llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    return llm, embeddings

def connect_to_weaviate(weaviate_url, weaviate_api_key):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        skip_init_checks=True
    )
    return client

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store_and_retriever(docs, embeddings, client):
    vector_db = WeaviateVectorStore.from_documents(docs, embeddings, client=client)
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

def main():
    st.title("PDF RAG Application with Weaviate")
    st.write("Upload a PDF file and type your question to extract relevant chunks from the document.")

    env_vars = load_environment_variables()

    llm, embeddings = initialize_llm_and_embeddings(env_vars["gemini_api_key"])

    client = connect_to_weaviate(env_vars["weaviate_url"], env_vars["weaviate_api_key"])

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        
        docs = load_and_split_pdf(uploaded_file)

        retriever = create_vector_store_and_retriever(docs, embeddings, client)


        rag_chain = define_rag_chain(retriever, llm)

        
        question = st.text_input("Type your question about the document:")

        if question:
            st.write("Processing...")
            llm_response = rag_chain.invoke(question)

            st.subheader("Result:")
            st.write(llm_response)

if __name__ == "__main__":
    main()