from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.run_nables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Khởi tạo FastAPI NGAY LẬP TỨC để Render thấy cổng mạng mở
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Biến toàn cục để lưu trữ bộ não AI sau khi nạp xong
global_rag_chain = None

# 2. Hàm nạp dữ liệu (Chỉ chạy khi có người nhắn tin lần đầu)
def initialize_ai():
    global global_rag_chain
    if global_rag_chain is not None:
        return

    print("--- Bắt đầu nạp dữ liệu AI ngầm ---")
    loader = TextLoader("baucu_data.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    system_prompt = (
        "Bạn là trợ lý ảo tư vấn pháp luật về Bầu cử đại biểu Quốc hội và Hội đồng nhân dân tại Việt Nam. "
        "Sử dụng các thông tin được cung cấp dưới đây để trả lời câu hỏi. "
        "Nếu thông tin không có trong tài liệu, hãy nói đúng nguyên văn: 'Dạ, hiện tại tôi chưa có thông tin chính xác về vấn đề này. Mời bạn tìm kiếm thêm thông tin trực tiếp trên Cổng thông tin điện tử của Đảng bộ Phường tại địa chỉ: https://dangbo.phuongchanhhung.vn/' "
        "Trả lời ngắn gọn, súc tích.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    global_rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("--- AI đã sẵn sàng! ---")

# 3. Các cổng giao tiếp (Endpoints)
@app.get("/")
async def root():
    return {"message": "Chatbot Bau Cu is Online!"}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        initialize_ai() # Nạp dữ liệu nếu chưa nạp
        response = global_rag_chain.invoke(req.message)
        return {"reply": response}
    except Exception as e:
        print(f"Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
