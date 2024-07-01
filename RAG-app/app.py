import boto3
import streamlit as st
import os

# Bedrock and embedding related imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data ingestion related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector store related imports
from langchain.vectorstores import FAISS

# LLM models related imports
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

# Initialize BedrockEmbeddings for Titan model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def load_and_split_pdfs(directory):
    """
    Load and split PDF documents from a directory using PyPDFDirectoryLoader and RecursiveCharacterTextSplitter.
    Args:
        directory (str): Path to the directory containing PDF files.
    Returns:
        List of split documents.
    """
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store(docs):
    """
    Create and save a FAISS vector store from split documents.
    Args:
        docs (list): List of split documents.
    Returns:
        FAISS: Created FAISS index.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_claude_llm():
    """
    Create and return the Claude LLM model instance.
    """
    return Bedrock(
        model_id="ai21.j2-mid-v1",
        client=bedrock_client,
        model_kwargs={'maxTokens': 512}
    )

def get_llama2_llm():
    """
    Create and return the LLaMA 2 LLM model instance.
    """
    return Bedrock(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock_client,
        model_kwargs={'max_gen_len': 512}
    )

# Prompt template for QA
prompt_template = """
Human: Incorporate the provided context to deliver a detailed response
to the question below. Your answer should be thorough, comprising a 
minimum of 250 words with comprehensive explanations. If uncertain,
please acknowledge your uncertainty rather than speculating.
<context>
{context}
</context>
Question: {question}
Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    """
    Perform QA using RetrievalQA with specified LLM and vector store.
    Args:
        llm (Bedrock): Instance of Bedrock LLM.
        vectorstore_faiss (FAISS): Instance of FAISS vector store.
        query (str): Query to be processed.
    Returns:
        Result of QA as a string.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def load_faiss_index():
    """
    Load the FAISS index if it exists.
    Returns:
        FAISS: Loaded FAISS index or None if the index doesn't exist.
    """
    if os.path.exists("faiss_index"):
        try:
            return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return None
    else:
        return None

def main():
    """
    Main function to setup Streamlit app for interacting with given documents using AWS Bedrock.
    """
    # Set up the Streamlit page
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with your PDF document using AWS Bedrock")

    # Initialize session state to store the FAISS index
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = load_faiss_index()

    # Create a text input for user questions
    user_question = st.text_input("Ask a question from your documents")

    # Create a sidebar for vector store updates
    with st.sidebar:
        st.title("Please upload a PDF and update vectors first.")
        
        # Add a file uploader for PDF files
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        # Add a button to update vectors
        if st.button("Update Vectors"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    # Save the uploaded file temporarily
                    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the uploaded file
                    docs = load_and_split_pdfs("temp")
                    st.session_state.faiss_index = create_vector_store(docs)
                    
                    # Remove the temporary file
                    os.remove(os.path.join("temp", uploaded_file.name))
                    
                    st.success("Vector store updated successfully!")
            else:
                st.error("Please upload a PDF file before updating vectors.")

    # Function to handle LLM processing
    def process_llm(llm_func):
        if st.session_state.faiss_index is not None:
            if user_question:
                with st.spinner("Processing..."):
                    try:
                        llm = llm_func()
                        st.write(get_response_llm(llm, st.session_state.faiss_index, user_question))
                        st.success("Done")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a question before processing.")
        else:
            st.warning("FAISS index not loaded. Please upload a PDF and update vectors first.")

    # Add a button to use Claude LLM
    if st.button("Use Claude"):
        process_llm(get_claude_llm)

    # Add a button to use Llama2 LLM
    if st.button("Use Llama2"):
        process_llm(get_llama2_llm)

if __name__ == "__main__":
    # Create a temporary directory to store uploaded files
    os.makedirs("temp", exist_ok=True)
    main()

