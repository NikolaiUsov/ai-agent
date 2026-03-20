import os
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.vectorstores import FAISS

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = str(BASE_DIR / "docs" / "knowledge_sources") 
faiss_index_path = str(BASE_DIR / "faiss_index")
chunk_size=500
chunk_overlap=100
separators=["\n\n", "\n", ". ", " "]

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Модель загружена!")
# Смотрим размерность эмбеддингов модели
test_embedding = embeddings_model.embed_query("проверка размерности")
print(f"📏 Размерность: {len(test_embedding)}")


# Разделение текста
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=separators
)

# Проверяем наличие папки
if not Path(KNOWLEDGE_DIR).exists():
    print(f"Папка {KNOWLEDGE_DIR} не найдена!")
    docs = []
else:
    # Загружаем все PDF
    loader = DirectoryLoader(
        path=KNOWLEDGE_DIR,
        glob="*.pdf",
        use_multithreading=True,
        loader_cls=PDFPlumberLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"Загружено страниц: {len(docs)}")
    # Статистика по файлам
    sources = {}
    for doc in docs:
        fname = Path(doc.metadata.get("source", "unknown")).name
        sources[fname] = sources.get(fname, 0) + 1
    print(f"📄 Файлов: {len(sources)}")
    for fname, pages in sorted(sources.items()):
        print(f"   • {fname}: {pages} стр.")

if not docs:
    raise SystemExit(
        "Нет загруженных документов для построения индекса. "
        "Проверьте, что в 'docs/knowledge_sources' есть PDF-файлы."
    )

# Проверяем, существует ли папка с индексом и есть ли в ней файлы
if os.path.exists(faiss_index_path) and os.path.isfile(os.path.join(faiss_index_path, "index.faiss")):
    print("✅ Индекс найден!")
else:
    print(f"\n🆕 Индекс не найден. Создаём новый в '{faiss_index_path}'...")

    # Создание чанков с метаданными
    chunks = text_splitter.split_documents(docs)
    print(f"🔪 Чанков создано: {len(chunks)}")

    # Создание FAISS индекса
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    print(f"📊 Количество векторов: {vectorstore.index.ntotal}")
    vectorstore.save_local(faiss_index_path)
    print("✅ Индекс сохранён и готов к использованию!")

