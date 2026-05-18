# AI Document Chatbot

## Project Overview
This project is an AI-based document chatbot
Users can upload documents like:
PDF
 DOCX
TXT

The chatbot reads the document and answers questions based on the uploaded file
## Workflow Diagram
```
Upload Document
        ↓
Read Document Text
        ↓
Store Information
        ↓
User Asks Question
        ↓
Find Relevant Content
        ↓
Generate Answer
```
##  Diagram
```
User
  ↓
Upload File
  ↓
Text Extraction
  ↓
Vector Database
  ↓
AI Model
  ↓
Answer Generation
```
## Technologies Used
 Python
 Streamlit
 LangChain
FAISS
 Groq API
PyPDF
python-docx
## How Project Works

### Step 1: Upload Document
User uploads a file
### Step 2: Read Text
The chatbot reads text from the uploaded document.
### Step 3: Store Information
Document information is stored for searching
### Step 4: Ask Questions
User asks questions.
### Step 5: Generate Answer
The chatbot finds related information and gives an answer
## Features
Upload documents
Ask questions
Get answers from document
Supports PDF, DOCX, TXT
## Run Project
Install libraries:
```
pip install -r requirements.txt
```
Run project:
```
python -m streamlit run app.py
```
## Future Improvements
 Chat history
 Better UI
 More document support
 Faster responses

 python -m streamlit run app.py --server.port 8502 --server.headless true