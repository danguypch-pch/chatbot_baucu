from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- CẤU HÌNH API KEY ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAF7Nxt1DMLYi4dQKgEIUNPedRjoeDzngY" # Thay bằng API Key của bạn

# --- 1. XÂY DỰNG HỆ THỐNG RAG (SỬ DỤNG LCEL HIỆN ĐẠI) ---
print("Đang nạp dữ liệu bầu cử...")
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
    "Nếu thông tin không có trong tài liệu, hãy nói rằng 'Theo dữ liệu hiện tại, tôi không có thông tin chính xác về vấn đề này. Bạn vui lòng liên hệ cơ quan có thẩm quyền.' "
    "Tuyệt đối không bịa đặt, không đưa ra quan điểm chính trị cá nhân. Trả lời ngắn gọn, súc tích và dễ hiểu.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Hàm phụ trợ để nối các đoạn tài liệu tìm được
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Tạo RAG Chain bằng kiến trúc LCEL (Không cần dùng module langchain.chains bị lỗi nữa)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 2. XÂY DỰNG FASTAPI SERVER ---
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

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Lấy câu trả lời trực tiếp
        response = rag_chain.invoke(req.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)