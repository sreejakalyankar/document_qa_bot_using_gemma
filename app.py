import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # vectorstore db
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile

load_dotenv()

def vector_embedding(uploaded_files):
    # Check if the vector store is already created
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Temporary storage for document loading
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load PDF document from temporary file
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())

            # Delete the temporary file
            os.remove(temp_file_path)

        # Split the documents into chunks and create the vector store
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector Store DB is Ready")

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Set up Streamlit UI
st.title("Document QA Bot using Gemma Model")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# File uploader to upload PDF documents
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Button to create vector store
if st.button("Creating Vector Store") and uploaded_files:
    vector_embedding(uploaded_files)

# Input for user query
prompt1 = st.text_input("What you want to ask from the documents?")

# "Submit" button to trigger the response generation
if st.button("Submit") and prompt1:
    if "vectors" in st.session_state:
        # Create document and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Query the chain and display response
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])
        
        # Show document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please create the vector store first by uploading files and clicking 'Creating Vector Store'.")



