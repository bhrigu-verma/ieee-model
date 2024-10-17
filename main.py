# File: main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pdf_processing import process_pdfs, generate_questions, user_input
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Question(BaseModel):
    question: str

@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    processed_text = await process_pdfs(files)
    questions = generate_questions(processed_text)
    return {"message": "PDFs processed successfully", "questions": questions}

@app.post("/ask-question")
async def ask_question(question: Question):
    answer = user_input(question.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# File: pdf_processing.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import re

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def process_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        content = await pdf.read()
        pdf_reader = PdfReader(BytesIO(content))
        for page_num, page in enumerate(pdf_reader.pages, 1):
            content = page.extract_text()
            text += f"[PDF: {pdf.filename}, Page: {page_num}]\n{content}\n\n"
    
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. Include citations for your answer in the format [PDF: filename, Page: number].
    strictly provide citations below whether asked or not in each response
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_citations(response):
    citations = re.findall(r'\[PDF: (.*?), Page: (\d+)\]', response)
    return citations

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    
    answer = response["output_text"]
    citations = extract_citations(answer)
    
    return {"answer": answer, "citations": citations}

def generate_questions(text):
    prompt = f"""
    Based on the following text, generate 5 relevant questions that could be asked about the content:

    {text[:10000]}  # Limiting to first 10000 characters to avoid token limit

    Generate 5 questions:
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    questions = response.text.split('\n')
    return [q.strip() for q in questions if q.strip()]

# File: streamlit_app.py
import streamlit as st
import requests

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                files = [("files", file) for file in pdf_docs]
                response = requests.post("http://localhost:8000/upload-pdfs", files=files)
                if response.status_code == 200:
                    st.session_state['processed'] = True
                    st.session_state['questions'] = response.json()['questions']
                    st.success("Done")
                else:
                    st.error("Error processing PDFs")

    if 'processed' in st.session_state and st.session_state['processed']:
        st.write("Suggested questions:")
        for i, question in enumerate(st.session_state['questions'], 1):
            if st.button(f"Q{i}: {question}"):
                ask_question(question)

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        ask_question(user_question)

def ask_question(question):
    response = requests.post("http://localhost:8000/ask-question", json={"question": question})
    if response.status_code == 200:
        answer_data = response.json()
        st.write("Answer:", answer_data['answer']['answer'])
        if answer_data['answer']['citations']:
            st.write("Citations:")
            for pdf_name, page_num in answer_data['answer']['citations']:
                st.write(f"- {pdf_name}, Page {page_num}")
    else:
        st.error("Error getting answer")

if __name__ == "__main__":
    main()