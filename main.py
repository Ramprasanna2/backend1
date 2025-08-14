from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = Ollama(model="mistral")  # Connect to local Mistral via Ollama

print("[Loading Chroma Vector Store...]")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # same model used during indexing

# Assuming your Chroma data already exists in ./chroma_db
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="patient_documents"
)



class PromptRequest(BaseModel):
    prompt: str
    id:str

metadata={}



@app.post("/generatePatientDetails")
async def generate_text(request: PromptRequest):
    system_prompt="""
You are a concise, factual assistant.
- Always be warm, gentle, and reassuring in tone.
- Use only the relevant context provided.
- If the context does not directly relate to the user’s question, ignore it.
- Do not add unrelated information.
- Do not speculate or assume.
- Repeat important information gently if needed.
- Answer in the shortest, most direct way possible while being accurate.
Your goal: Provide helpful, caring, and precise answers while keeping the user feeling safe, respected, and understood."""
    user_prompt = request.prompt
    user_id=request.id
    
    print(user_prompt)
    # 1. Retrieve relevant documents from Chroma filtered by patient_id
    results = vectorstore.similarity_search(
        query=user_prompt,
        k=5,  # fetch top 3 relevant chunks
        filter={"patient_id": user_id}
    )
    # 2. Combine the content of retrieved chunks into context
    context = "\n".join([doc.page_content for doc in results])
    full_prompt=f"""system:{system_prompt}
   context:{context}
   user:{user_prompt}
   AI:
    """
    print(results)


    # 4. Call the LLM
    response = llm.invoke(full_prompt)

    return {"response": response}

def store_text(text: str, metadata: dict=None, persist_dir: str = "./chroma_db"):
    if len(text) <= 450:  # short text threshold
        chunks = [text]
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metadatas = [metadata for _ in chunks]
   
    db = Chroma(
        collection_name="patient_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
 
    db.add_texts(chunks, metadatas=metadatas)
    db.persist()
    return len(chunks)


@app.post("/store_patient_data")
async def store_patient_data(request:Request):
    data =await request.json()
    text = data.get("text")
    print(text)
    metadata = data.get("metadata")  # expected to be a dict
    print(text)
    print(metadata)
    if not text or not metadata:
        return {"error": "Both 'text' and 'metadata' fields are required."}

    num_chunks = store_text(text, metadata)
    return {"message": f"Stored {num_chunks} chunks with metadata."}

@app.post("/delete_patient_data")
async def deleteData(request:Request):
    data = await request.json()
    metadata = data.get("metadata")  # expected to be a dict

    if not metadata:
        return {"error": "Field 'metadata' is required."}

    # Delete documents from Chroma based on metadata filter
    deleted_count = vectorstore._collection.delete(where=metadata)
    vectorstore.persist()  # persist changes

    return deleted_count

if __name__ == "__main__":
    import uvicorn
    # Run with optimized settings
    uvicorn.run(
        "main:app",  # Replace 'main' with your filename
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker to avoid model loading multiple times
        loop="asyncio"
    )
    

