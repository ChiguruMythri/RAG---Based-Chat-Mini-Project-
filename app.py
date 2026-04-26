import streamlit as st
import os
import faiss
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

# Modern LangChain Standard Imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# API Key Loading
try:
    from secret_api_keys import huggingface_api_key
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
except ImportError:
    st.error("Missing secret_api_keys.py file!")

def process_document(source_type, data):
    text = ""
    if source_type == "PDF":
        pdf = PdfReader(BytesIO(data.read()))
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif source_type == "Link":
        loader = WebBaseLoader([url for url in data if url.strip()])
        text = " ".join([d.page_content for d in loader.load()])
    elif source_type == "Text":
        text = data
    elif source_type == "DOCX":
        doc = Document(BytesIO(data.read()))
        text = "\n".join([p.text for p in doc.paragraphs])

    if not text: raise ValueError("No text extracted.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(chunks, embeddings)

def ask_llama(vectorstore, question):
    # Base LLM Setup
    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
    )

    # Conversational Fix: Using ChatHuggingFace wrapper
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the context: {context}"),
        ("human", "{input}")
    ])

    # Modern RAG Pipeline
    combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    response = rag_chain.invoke({"input": question})
    return response["answer"]

def main():
    st.set_page_config(page_title="Advanced RAG System", page_icon="🤖")
    st.title("🚀 Advanced RAG Q&A System (Llama-3)")
    st.markdown("---")

    st.sidebar.header("📁 Data Source")
    source = st.sidebar.selectbox("Select Source", ["PDF", "Link", "Text", "DOCX"])
    
    user_input = None
    if source == "Link":
        link = st.sidebar.text_input("Enter URL:")
        user_input = [link] if link else None
    elif source == "Text":
        user_input = st.text_area("Paste text here:")
    else:
        user_input = st.sidebar.file_uploader(f"Upload {source}")

    if st.sidebar.button("Build Knowledge Base"):
        if user_input:
            with st.spinner("Processing..."):
                try:
                    st.session_state.vs = process_document(source, user_input)
                    st.sidebar.success("✅ Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if "vs" in st.session_state:
        query = st.text_input("Ask a question:")
        if st.button("Get Answer") and query:
            with st.spinner("Llama-3 is thinking..."):
                try:
                    answer = ask_llama(st.session_state.vs, query)
                    st.write(answer)
                except Exception as e:
                    st.error(f"AI Error: {e}")

if __name__ == "__main__":
    main()