import streamlit as st
from src.helper_rag import (
    load_environment_variables,
    initialize_llm_and_embeddings,
    load_and_split_pdf,
    create_vector_store_and_retriever,
    define_rag_chain
)

def main():
    st.title("PDF RAG App")
    st.write(
        "Upload a PDF and ask questions – this app uses Chroma as its vector store "
        "to retrieve the most relevant document chunks."
    )

    env_vars = load_environment_variables()
    llm, embeddings = initialize_llm_and_embeddings(env_vars["gemini_api_key"])

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if not uploaded_file:
        return

    docs = load_and_split_pdf(uploaded_file)

    retriever = create_vector_store_and_retriever(
        docs,
        embeddings,
        persist_directory="chroma_db"
    )

    rag_chain = define_rag_chain(retriever, llm)

    question = st.text_input("Type your question about the document:")
    if not question:
        return

    st.write("Processing…")
    answer = rag_chain.invoke(question)

    st.subheader("Result:")
    st.write(answer)


if __name__ == "__main__":
    main()
