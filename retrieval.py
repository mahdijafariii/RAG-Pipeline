from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
def get_directory_documents(path : str):

    # checking if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')

    files_names = os.listdir(path)
    readme_files_names = [file for file in files_names if file.endswith('.txt') == True]

    if len(readme_files_names) == 0:
        raise ValueError(f'path : {path} does not contain any text files')

    documents = []
    for file in readme_files_names:
        loader = TextLoader(os.path.join(path, file), encoding = 'UTF-8')
        document = loader.load()
        documents.append(document[0])

    return documents

class Retrieval:

    def __init__(self,
                 embedding_model_name,
                 documents_address,
                 vector_db_address = None):

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

            from uuid import uuid4
            import faiss

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

    def get_docs(self, path: str) :
        return get_directory_documents(path)


    def __call__(self,
                 inp_text,
                 n_returned_docs = 3):

        contexts = self.db.similarity_search(inp_text, k = n_returned_docs)
        outputs = [context.page_content for context in contexts]

        return outputs

    def add_document(self):
        raise NotImplementedError
