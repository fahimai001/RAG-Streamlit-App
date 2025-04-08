# PDF RAG Application with FAISS

## Overview

This Streamlit-based application enables users to upload a PDF document, ask questions about its content, and retrieve relevant text snippets using a Retrieval-Augmented Generation (RAG) pipeline. It leverages FAISS for vector storage and Google Gemini for generating responses.

![RAG APPLICATION DEMO](D:\JMM_Technologies\RAG_Based_App\rag.png)


## Features

- **PDF Upload**: Upload a PDF file to the application.
- **Question Answering**: Ask questions about the uploaded PDF document.
- **Relevant Chunk Extraction**: The application retrieves and displays the most relevant chunks of text from the document that answer the question.
- **Powered by FAISS and Google Gemini**: Uses FAISS for efficient vector storage and retrieval, and Google Gemini for generating responses.

## Why FAISS?

FAISS (Facebook AI Similarity Search) offers several advantages:

- **Performance**: Extremely fast similarity search for dense vectors
- **Scalability**: Efficiently handles millions of vectors
- **Local Operation**: No need for external servers or cloud services
- **Zero Additional Costs**: Free to use, no subscription required
- **Privacy**: All data stays on your machine

## Prerequisites

Before running the application, ensure you have the following:

1. **Python 3.8 or higher**: The application is built using Python.
2. **API Keys**:
   - **Google Gemini API Key**: Required for using the Gemini model.
3. **Environment Variables**: Store your API key in a `.env` file in the root directory of the project.

## Setup

### Environment Variables

Store API keys securely in a `.env` file at the project's root directory:

```
GEMINI_API_KEY=your_gemini_api_key
```

### Clone the Repository

```bash
git clone https://github.com/fahimai001/RAG-Streamlit-App.git
cd pdf-rag-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## How to Use

1. **Launch the Application**: Open your browser and navigate to `http://localhost:8501`.
2. **Upload a PDF**: Click on "Choose a PDF file" to upload your document.
3. **Ask a Question**: Type your question in the text input field.
4. **View Results**: The application will display the most relevant information extracted from the document.

## Try It Yourself!

Experiment with different types of PDFs and questions to see how the application performs. Here are some suggestions:

- Upload a research paper and ask about specific findings
- Upload a manual and ask about particular procedures
- Upload a report and ask for key statistics

## Contributing

We welcome contributions to improve this application!