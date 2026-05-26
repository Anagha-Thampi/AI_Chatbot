from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os
import streamlit as st
import shutil

st.title("RAG AI CHATBOT")

files=st.file_uploader("Upload Files: ",accept_multiple_files=True,type="pdf")

if files:
    if os.path.exists("temp"):

        shutil.rmtree("temp")

    os.makedirs("temp", exist_ok=True)
    doc = []
    for file in files:
        path = f"temp/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(path)
        d = loader.load()
        doc.extend(d)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )

    docs=text_splitter.split_documents(doc)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
       search_kwargs={"k": 3}
    )

    os.environ["GROQ_API_KEY"] =os.environ.get("GROQ_API_KEY")

    llm = ChatGroq(
       model_name="llama-3.1-8b-instant"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q=st.chat_input("Enter question: ")

    if q:
        retrieved_docs = retriever.invoke(q)
        with st.chat_message("user"):
            st.write(q)

        context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
        )

        history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.chat_history
        ])

        prompt = f"""
        Answer the question using ONLY the context below.
        Conversation History:
            {history}
        Context:
        {context}
        Question:
        {q}
        If the answer is not in the context, say:
        "I could not find the answer in the provided document."
        """
        response = llm.invoke(prompt)
        with st.chat_message("assistant"):
            st.write(response.content)
            st.subheader("Source Chunks")

            for i, doc in enumerate(retrieved_docs):

                source = doc.metadata.get(
                        "source",
                        "Unknown"
                        )

                page = doc.metadata.get(
                    "page",
                    "N/A"
                )

            with st.expander(
            f"Source {i+1} — {source} (Page {page})"
            ):

                st.write(doc.page_content)

        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": q
        })

        # Save assistant response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.content
        }) 
