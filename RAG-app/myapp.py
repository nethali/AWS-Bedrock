import boto3
import streamlit as st

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
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Initialize BedrockEmbeddings for Titan model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def data_ingestion():
    """
    Function to load documents from a directory using PyPDFDirectoryLoader and split them using RecursiveCharacterTextSplitter.
    Returns:
        List of split documents.
    """
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """
    Function to create and save FAISS vector store from documents.
    Args:
        docs (list): List of documents.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    """
    Function to create and return the Claude LLM model instance.
    """
    return Bedrock(
        model_id="ai21.j2-mid-v1",
        client=bedrock_client,
        model_kwargs={'maxTokens': 512}
    )

def get_llama2_llm():
    """
    Function to create and return the LLaMA 2 LLM model instance.
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
    Function to perform QA using RetrievalQA with specified LLM and vector store.
    Args:
        llm (Bedrock): Instance of Bedrock LLM.
        vectorstore_faiss (FAISS): Instance of FAISS vector store.
        query (str): Query to be processed.
    Returns:
        Result of QA as a dictionary.
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

def main():
    """
    Main function to setup Streamlit app for interacting with given documents using AWS Bedrock.
    """
    st.set_page_config(page_title="Chat with OpenShift AI")
    
    st.header("Chat with given PDF documents using AWS Bedrock")

    user_question = st.text_input("Ask a question from your documents")

    with st.sidebar:
        st.title("Create or update Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Use Claude"):
        with st.spinner("Processing..."):
            try:
                # Load FAISS index with dangerous deserialization allowed
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_claude_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")
            except ValueError as e:
                st.error(f"An error occurred: {e}")

    if st.button("Use Llama2"):
        with st.spinner("Processing..."):
            try:
                # Load FAISS index with dangerous deserialization allowed
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama2_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")
            except ValueError as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

