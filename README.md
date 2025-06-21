# 🧠 AI News Research Tool 📰

Welcome to the **AI News Research Tool**, a Streamlit-powered web app that allows you to **analyze multiple news articles** and ask AI questions about them using **LangChain**, **OpenAI GPT**, and **FAISS vector search**.

---

## 🚀 Features

- 🔗 **URL Input**: Paste up to 3 news article URLs
- 📄 **Automatic Chunking**: Splits articles into AI-readable sections
- 🧠 **Embeddings with OpenAI**: Converts text into smart vectors
- 🔍 **Ask Questions**: Query the articles and get AI-generated answers with sources
- 🗂️ **Vector Store**: Built with FAISS for fast retrieval

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [LangChain](https://www.langchain.com/)
- **LLM**: `gpt-3.5-turbo` via [OpenAI](https://openai.com/)
- **Embeddings**: `OpenAIEmbeddings`
- **Vector DB**: FAISS
- **Text Loader**: `UnstructuredURLLoader`

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vaibhavknight18/equityAnalyzerLLM.git
cd equityAnalyzerLLM
pip install -r requirements.txt
OPENAI_API_KEY=your_openai_key_here
streamlit run app.py
equityAnalyzerLLM/
├── app.py               # Streamlit application
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignore sensitive/local files
├── .env                 # API keys (not pushed to repo)
└── README.md            # Project overview
