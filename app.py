from langchain_community.vectorstores import FAISS      
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Load the .env file
load_dotenv()

# Create the ChatGoogleGenerativeAI object
llm:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI( model="models/gemini-1.5-flash")

file_path : str= "./Generative-AI-Foundations-in-Python.pdf"

loader : PyPDFLoader = PyPDFLoader(file_path)

docs : str = loader.load()

document = RecursiveCharacterTextSplitter(chunk_size = 2000 , chunk_overlap = 200).split_documents(documents=docs)

vector_store : FAISS = FAISS.from_documents( document , GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vector_store.as_retriever()


# Define the system prompt as a single string
system_prompt = (
    "You are an assistant for question answering task. "
    "Use the following pieces of retrieved context to answer the question. "
    "Do not hallucinate if you do not know the answer. Just say you do not know. "
    "Use three sentences maximum and keep the answer concise and to the point.\n\n"
    "{context}"
)

# Create the ChatPromptTemplate object
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

ques_ans_chain = create_stuff_documents_chain(prompt=prompt   , llm=llm)
rag_chain = create_retrieval_chain(retriever , ques_ans_chain)

results = rag_chain.invoke({"input": input("Ask a question: ")})

print(results["answer"])
