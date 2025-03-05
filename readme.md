# PDF RAG Application with Weaviate

This is a Streamlit-based application that allows users to upload a PDF document, ask questions about its content, and retrieve relevant chunks of text using a Retrieval-Augmented Generation (RAG) pipeline powered by Weaviate and Google Gemini.

## Features

- **PDF Upload**: Upload a PDF file to the application.
- **Question Answering**: Ask questions about the uploaded PDF document.
- **Relevant Chunk Extraction**: The application retrieves and displays the most relevant chunks of text from the document that answer the question.
- **Powered by Weaviate and Google Gemini**: Uses Weaviate for vector storage and retrieval, and Google Gemini for generating responses.

## Prerequisites

Before running the application, ensure you have the following:

1. **Python 3.8 or higher**: The application is built using Python.
2. **API Keys**:
   - **Google Gemini API Key**: Required for using the Gemini model.
   - **Weaviate API Key**: Required for connecting to the Weaviate cloud.
3. **Environment Variables**: Store your API keys in a `.env` file in the root directory of the project.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fahimai001/RAG-Streamlit-App.git
   cd pdf-rag-app