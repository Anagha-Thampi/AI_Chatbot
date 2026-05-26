from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os
import logging

logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

loader=PyPDFLoader("Transport Layer- Services.pdf")
doc1=loader.load()

l=PyPDFLoader("Network_Topology_Presentation.pdf")
doc2=l.load()
logging.info("PDFs loaded successfully")

doc=[]
doc.extend(doc2)
doc.extend(doc1)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs=text_splitter.split_documents(doc)
logging.info(f"Total chunks created: {len(docs)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

if os.path.exists("./chroma_db"):

    logging.info("Loading existing ChromaDB")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
else:
    logging.info("Creating new ChromaDB")
    vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
    )

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)



llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key="enter your api key"
)


chat_history=[]

while(True):

    q=input("\nEnter question: ")
    logging.info(f"User Question: {q}")

    if q.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(q)
    logging.info(f"Retrieved {len(retrieved_docs)} chunks")
    for d in retrieved_docs:
        logging.info(f"Source: {d.metadata}")
    
    print("RETRIEVED CHUNKS:\n")

    for i, d in enumerate(retrieved_docs):

        print(f"\nChunk {i+1}:\n")
        print(d.page_content)
        print("-------------------------\n")

    context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
    )
    
    history = "\n".join(chat_history)

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

    logging.info("Sending prompt to LLM")
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        logging.error(f"LLM Error: {e}")
        print("Error occurred")
        continue
    
    print("\nANSWER: ")
    print(response.content)
    logging.info("Response generated successfully")

    chat_history.append(f"User: {q}")
    chat_history.append(f"Assistant: {response.content}")