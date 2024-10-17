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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text along with metadata (PDF name and page number)
def get_pdf_text_with_metadata(pdf_docs):
    text_with_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text_with_metadata.append({
                "text": page.extract_text(),
                "metadata": {
                    "pdf_name": pdf.name,
                    "page_number": page_num + 1
                }
            })
    return text_with_metadata


# Function to split text into chunks
def get_text_chunks(text_with_metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_metadata = []
    for entry in text_with_metadata:
        chunks = text_splitter.split_text(entry["text"])
        for chunk in chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": entry["metadata"]
            })
    return chunks_with_metadata


# Function to store text chunks with metadata in FAISS
def get_vector_store_with_metadata(text_chunks_with_metadata):
    texts = [chunk["text"] for chunk in text_chunks_with_metadata]
    metadata = [chunk["metadata"] for chunk in text_chunks_with_metadata]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")


# Function to create a conversational chain using the AI model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to format and display citations
def format_citations(metadata):
    citations = []
    for data in metadata:
        citations.append(f"PDF: {data['pdf_name']}, Page: {data['page_number']}")
    return citations


# Function to handle user input and display response with citations
def user_input_with_citations(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])
    metadata = [doc.metadata for doc in docs]  # Metadata for citations

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])
    st.write("Citations: ", format_citations(metadata))


# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
             
            print("Processing PDFs...")
            raw_text_with_metadata = get_pdf_text_with_metadata(pdf_docs)
            text_chunks_with_metadata = get_text_chunks(raw_text_with_metadata)
            get_vector_store_with_metadata(text_chunks_with_metadata)
            st.success("Done")
                
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input_with_citations(user_question)


if __name__ == "__main__":
    main()
