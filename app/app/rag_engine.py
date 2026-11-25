import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGEngine:
    def __init__(self, llm_model="qwen2.5:3b", embed_model="nomic-embed-text"):
        self.ollama_url = "http://localhost:11434"
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.answer_translator = None

    def load_and_build(self, file_path, progress_callback=None):
        """Load PDF and build RAG pipeline"""
        if progress_callback:
            progress_callback("📂 در حال بارگذاری فایل...")

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        if progress_callback:
            progress_callback(f"✅ فایل خوانده شد ({len(documents)} صفحه). در حال خرد کردن...")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if progress_callback:
            progress_callback(f"✅ {len(chunks)} تکه ساخته شد. در حال ساخت بردارها...")

        embeddings = OllamaEmbeddings(model=self.embed_model, base_url=self.ollama_url)
        self.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        if progress_callback:
            progress_callback("✅ پایگاه دانش آماده است. در حال راه‌اندازی مدل...")

        self._build_chains()

        if progress_callback:
            progress_callback("✅ سیستم آماده پرسش است!")

        return True

    def _build_chains(self):
        """Build chains with balanced prompts"""
        llm = ChatOllama(model=self.llm_model, base_url=self.ollama_url, temperature=0.1)

        # === Main RAG prompt (English, balanced - not too strict) ===
        rag_prompt = ChatPromptTemplate.from_template(
            """You are a mechanical engineer. Answer the question using ONLY the context below.

Rules:
- Find the exact value/number in the context that matches the question.
- The context may be a table. Match the item name to its grade value.
- If the item is truly not mentioned anywhere in the context, write: NOT_FOUND
- Otherwise, answer concisely with the exact value.

Context:
{context}

Question: {question}

Answer:"""
        )

        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        # === Answer translator (English -> Persian) ===
        a_prompt = ChatPromptTemplate.from_template(
            """Translate this English answer to Persian. Keep technical terms and numbers exactly as they are. Output ONLY the Persian translation.

English: {text}
Persian:"""
        )
        self.answer_translator = (
            {"text": RunnablePassthrough()}
            | a_prompt
            | llm
            | StrOutputParser()
        )

    def _is_persian(self, text):
        return bool(re.search(r'[آ-ی]', text))

    def _dual_search(self, question):
        """Search with BOTH original question and a keyword-based English fallback"""
        # Search 1: original question (works for Persian too, since nomic is multilingual)
        docs_fa = self.retriever.invoke(question)

        # Search 2: extract key technical terms and search in English
        # Build a simple English query from the original
        docs_en = docs_fa
        if self._is_persian(question):
            # Create keyword query: keep numbers/grades, add common English terms
            keywords = re.findall(r'[A-Za-z0-9.,]+', question)
            en_query = " ".join(keywords) + " " + question
            docs_en = self.retriever.invoke(en_query)

        # Merge and deduplicate
        seen = set()
        merged = []
        for doc in docs_fa + docs_en:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged[:8]

    def ask(self, question, progress_callback=None):
        """Ask with dual search + direct LLM answering"""
        is_persian = self._is_persian(question)

        if progress_callback:
            progress_callback("🔍 جستجوی دوگانه در سند...")

        # Step 1: Dual retrieval
        docs = self._dual_search(question)
        context_text = "\n\n".join([d.page_content for d in docs])

        if progress_callback:
            progress_callback(f"✅ {len(docs)} تکه مرتبط پیدا شد.")

        # Step 2: Answer in English directly from context
        if progress_callback:
            progress_callback("🤔 مدل در حال استخراج پاسخ...")

        answer_en = self.rag_chain.invoke({"context": context_text, "question": question})

        # Step 3: Handle NOT_FOUND
        if "NOT_FOUND" in answer_en or "not found" in answer_en.lower():
            return "❌ این اطلاعات در سند ارائه‌شده وجود ندارد."

        # Step 4: Translate to Persian if needed
        if is_persian:
            if progress_callback:
                progress_callback("🔄 ترجمه پاسخ به فارسی...")
            try:
                answer_fa = self.answer_translator.invoke(answer_en)
                if len(answer_fa) >= 5 and re.search(r'[آ-ی]', answer_fa):
                    return answer_fa
                return answer_en
            except Exception:
                return answer_en

        return answer_en

    def get_debug_info(self, question):
        """Return retrieved chunks for the debug panel"""
        docs = self._dual_search(question)
        return docs