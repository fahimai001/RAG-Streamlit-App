# PDF RAG Application with Weaviate

# Overview

This Streamlit-based application enables users to upload a PDF document, ask questions about its content, and retrieve relevant text snippets using a Retrieval-Augmented Generation (RAG) pipeline. It leverages Weaviate for vector storage and Google Gemini for generating responses.

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

# Environment Variables

Store API keys securely in a .env file at the project's root directory:

GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
WEAVIATE_API_KEY=your_weaviate_api_key

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fahimai001/RAG-Streamlit-App.git
   cd pdf-rag-app

# Install Dependencies

pip install -r requirements.txt

# Run the Application

pip install -r requirements.txt