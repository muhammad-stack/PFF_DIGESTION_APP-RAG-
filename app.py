import streamlit as st
from langchain_community.vectorstores import FAISS      
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Load environment variables
load_dotenv()

# Initialize language model and retrieval chain
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

# Define prompt template for concise responses
system_prompt = (
    "You are an assistant for question answering task. "
    "Use the following pieces of retrieved context to answer the question. "
    "Do not hallucinate if you do not know the answer. Just say you do not know. "
    "Use three sentences maximum and keep the answer concise and to the point.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
ques_ans_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)

# Streamlit App Interface
st.title("Multi-Document PDF Q&A System")
st.write("Upload one or more PDF documents, then ask questions based on their content.")

# File Upload for multiple PDFs
uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# Initialize an empty list to hold document chunks
document_chunks = []

# Process each uploaded PDF file
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        # Split document into manageable chunks
        document_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(documents=docs))

    # Index all documents in the vector store
    vector_store = FAISS.from_documents(document_chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, ques_ans_chain)
    st.success(f"Successfully indexed {len(uploaded_files)} document(s).")

# User Question Input
user_question = st.text_input("Enter your question here:")

# Display answer if question is provided
if user_question:
    if not uploaded_files:
        st.warning("Please upload at least one PDF document to ask questions.")
    else:
        with st.spinner("Searching for an answer..."):
            results = rag_chain.invoke({"input": user_question})
            answer = results.get("answer", "I do not know the answer to that question.")
        st.write("**Answer:**", answer)
