import streamlit as st
from src.helper_rag import (
    load_environment_variables,
    initialize_llm_and_embeddings,
    load_and_split_pdf,
    create_vector_store_and_retriever,
    define_rag_chain
)

def main():
    st.title("PDF RAG Application with FAISS")
    st.write("Upload a PDF file and type your question to extract relevant information from the document.")

    env_vars = load_environment_variables()

    llm, embeddings = initialize_llm_and_embeddings(env_vars["gemini_api_key"])

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        
        docs = load_and_split_pdf(uploaded_file)

        retriever = create_vector_store_and_retriever(docs, embeddings)

        rag_chain = define_rag_chain(retriever, llm)

        question = st.text_input("Type your question about the document:")

        if question:
            st.write("Processing...")
            llm_response = rag_chain.invoke(question)

            st.subheader("Result:")
            st.write(llm_response)

if __name__ == "__main__":
    main()