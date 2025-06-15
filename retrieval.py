from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
import faiss
import os
from langchain.schema import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9\s\.]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_date(date_str: str) -> str | None:
    date_str = date_str.strip()

    try:
        dt = datetime.strptime(date_str, '%d %B %Y')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        pass

    match = re.match(r'(\d{4})s', date_str)
    if match:
        return match.group(1)

    match = re.match(r'^\d{4}$', date_str)
    if match:
        return date_str

    return None
def attach_metadata(text: str, title: str = None, date: str = None, category: str = None) -> dict:
    metadata = {}
    if title:
        metadata['title'] = title
    if date:
        metadata['date'] = date
    if category:
        metadata['category'] = category

    return {
        "text": text,
        "metadata": metadata
    }


def get_directory_documents(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')

    files_names = os.listdir(path)
    readme_files_names = [file for file in files_names if file.endswith('.txt')]

    if len(readme_files_names) == 0:
        raise ValueError(f'path : {path} does not contain any text files')

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in readme_files_names:
        loader = TextLoader(os.path.join(path, file), encoding='utf-8')
        docs = loader.load()

        for doc in docs:
            # پاک‌سازی متن
            cleaned_text = clean_text(doc.page_content)

            # استخراج اطلاعات از نام فایل (مثلاً title و تاریخ)
            filename_parts = file.replace(".txt", "").split("_")
            title = filename_parts[0] if len(filename_parts) > 0 else None
            raw_date = filename_parts[1] if len(filename_parts) > 1 else None
            normalized = normalize_date(raw_date) if raw_date else None

            # افزودن متادیتا
            meta_doc = attach_metadata(cleaned_text, title=title, date=normalized)

            # تبدیل به Document برای LangChain
            document = Document(page_content=meta_doc["text"], metadata=meta_doc["metadata"])

            # چانک کردن
            chunks = text_splitter.split_documents([document])
            documents.extend(chunks)

    return documents


class Retrieval:

    def __init__(self,
                 embedding_model_name,
                 documents_address,
                 vector_db_address=None):

        self.db_address = vector_db_address
        self.documents_address = documents_address

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name)

        if os.path.exists(self.db_address):
            print('>> HINT : loading documents embeddings ...')

            self.db = LangchainFAISS.load_local(
                folder_path=self.db_address,
                embeddings=self.embeddings, allow_dangerous_deserialization=True
            )

        else:

            print('>> HINT : embedding documents started ...')
            index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))

            self.db = LangchainFAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            documents = self.get_docs(path=self.documents_address)

            uuids = [str(uuid4()) for _ in range(len(documents))]

            self.db.add_documents(documents=documents, uuids=uuids)

            self.db.save_local(self.db_address)

    def get_docs(self, path: str):
        return get_directory_documents(path)

    def __call__(self,
                 inp_text,
                 n_returned_docs=3):

        contexts = self.db.similarity_search(inp_text, k=n_returned_docs)
        outputs = [context.page_content for context in contexts]

        return outputs

    def add_document(self):
        raise NotImplementedError
