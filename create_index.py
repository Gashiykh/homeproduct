from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

RAG_SEARCH_PROMPT_TEMPLATE = """
Используя следующие фрагменты извлеченного контекста, ответьте на вопрос полно и кратко.
Убедитесь, что ваш ответ полностью раскрывает вопрос, исходя из заданного контекста.

**ВАЖНО:**
Просто дайте ответ и никогда не упоминайте и не ссылайтесь на наличие доступа к внешнему контексту или информации в вашем ответе.
Если вы не можете определить ответ на основе предоставленного контекста, укажите «Я не знаю».

Вопрос: {question}
Контекст: {context}
"""

print("Loading & Chunking Docs...")
loader = TextLoader("./data/agency.txt", encoding="utf-8")
docs = loader.load()

doc_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
doc_chunks = doc_splitter.split_documents(docs)

print("Creating vector embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectorstore = Chroma.from_documents(doc_chunks, embeddings, persist_directory="db")

vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Test RAG chain...")
prompt = ChatPromptTemplate.from_template(RAG_SEARCH_PROMPT_TEMPLATE)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

rag_chain = (
    {"context": vectorstore_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "Название компании?"
result = rag_chain.invoke(query)
print(f"Question: {query}")
print(f"Answer: {result}")

