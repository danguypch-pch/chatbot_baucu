import os
import sys

# Mẹo cho cơ sở dữ liệu ChromaDB chạy được trên Render
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo App NGAY LẬP TỨC
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến để kiểm tra AI đã sẵn sàng chưa
rag_chain = None

def load_ai():
    global rag_chain
    if rag_chain is None:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        loader = TextLoader("baucu_data.txt", encoding="utf-8")
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        prompt = ChatPromptTemplate.from_template("Trả lời dựa trên tài liệu: {context}\nCâu hỏi: {input}")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        rag_chain = (
            {"context": vectorstore.as_retriever() | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def health_check():
    return {"status": "Online"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        load_ai() # Nạp AI khi có yêu cầu đầu tiên
        res = rag_chain.invoke(req.message)
        return {"reply": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
