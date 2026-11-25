import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def main():
    print("\n" + "="*60)
    print("🧠 سیستم RAG محلی و مستقیم (نسخه نهایی و بهینه) - مهندس رضا")
    print("="*60)
    
    file_path = input("\nمسیر فایل PDF را وارد کنید (یا فایل را اینجا Drag & Drop کنید): ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print("❌ فایل پیدا نشد! لطفاً مسیر را بررسی کنید.")
        return
        
    print(f"\n📂 در حال بارگذاری: {os.path.basename(file_path)}")
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print(f"✅ فایل با موفقیت خوانده شد ({len(documents)} صفحه).")
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return

    print("\n⏳ مرحله ۱: خرد کردن متن (Chunking) با سایز بزرگ‌تر برای حفظ جداول...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"✅ متن به {len(chunks)} بخش تقسیم شد.")

    print("\n⏳ مرحله ۲: ساخت پایگاه دانش (Embedding)...")
    print("   ⚠️ این مرحله ممکن است چند دقیقه طول بکشد. لطفاً صبر کنید.")
    
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        print("✅ پایگاه داده برداری با موفقیت ساخته شد.")
    except Exception as e:
        print(f"❌ خطا در ساخت پایگاه داده: {e}")
        print("   💡 مطمئن شوید برنامه Ollama باز است.")
        return

    print("\n⏳ مرحله ۳: راه‌اندازی مدل هوشمند (Qwen)...")
    llm = ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434", temperature=0.1)

    # پرامپت اصلاح‌شده و سخت‌گیرانه برای جلوگیری از توهم و ترجمه‌ی اشتباه
    prompt_template = """You are an expert mechanical engineer. Answer the user's question STRICTLY based on the provided English context.
- If the user asks in English, answer in English and extract exact numerical limits, values, and conditions.
- If the user asks in Persian, provide the answer in Persian, but keep technical terms and numbers exactly as they are in the text.
- If the exact answer is not in the context, you MUST say exactly: "این اطلاعات در سند ارائه‌شده وجود ندارد." (or "This information is not in the provided document" if asked in English).

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*60)
    print("🎉 آماده‌ام! سوال بپرسید (برای خروج exit)")
    print("="*60 + "\n")

    while True:
        try:
            question = input("❓ سوال: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if question.lower() == 'exit':
            break
        if not question:
            continue
            
        print("\n🔍 در حال جستجو... تکه‌های پیدا شده از سند (حالت دیباگ):")
        retrieved_docs = retriever.invoke(question)
        for i, doc in enumerate(retrieved_docs):
            print(f"--- تکه {i+1} (صفحه {doc.metadata.get('page', '?')}) ---")
            print(doc.page_content[:500].replace('\n', ' ') + "...\n")
        print("=========================\n")
        
        print("🤔 در حال فکر کردن و تولید پاسخ نهایی...\n")
        try:
            result = rag_chain.invoke(question)
            print(f"💡 پاسخ نهایی:\n{result}\n")
        except Exception as e:
            print(f"❌ خطا در تولید پاسخ: {e}")

if __name__ == "__main__":
    main()