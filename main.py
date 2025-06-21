import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("❌ OPENAI_API_KEY not found in environment. Please check your .env file.")
    st.stop()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=1000,
    api_key=openai_key  # ✅ Use `api_key` not `openai_api_key` if using langchain_openai
)

st.title("AI News Research Tool 🧠📰 ")
st.sidebar.title("News article URLS:")

url_labels = [
    "🚀 Drop the 🔗 to Your 1st Article Here!",
    "🧠 Got Another One? Paste the 2nd Article Link 📝",
    "🔍 Final Piece! Enter the 3rd Article URL 📰"
]
urls = [st.sidebar.text_input(label) for label in url_labels]
process_url_clicked = st.sidebar.button("🧠✨ Analyze My Articles")

file_path = "vectorstore_openai.pkl"
main_placefolder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    main_placefolder.text("🧠 **Running AI analysis on your URLs...** Sit back and relax! 🤖")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
    )
    main_placefolder.text("✂️ **Splitting the article into readable chunks...** Organizing content for better understanding! 📄")
    docs = text_splitter.split_documents(data)

    main_placefolder.text("🔍 **Creating AI embeddings for your articles...** Building a knowledge base! 📚")
    embeddings = OpenAIEmbeddings(api_key=openai_key)

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placefolder.text_input("🔍 **Ask a question about your articles:**")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)
            st.header("🧠 **AI's Answer:**")
            st.subheader(result["answer"])
