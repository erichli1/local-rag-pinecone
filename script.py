import os
import tiktoken
import pinecone
from tqdm.auto import tqdm
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from uuid import uuid4
from tkinter import filedialog

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

EXTENSIONS = [".pdf"]

# langchain
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
ENCODING_NAME = "cl100k_base"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# pinecone
INDEX_NAME = "basic"
MAX_CHUNK_SIZE = 400
UPSERT_BATCH_LIMIT = 100
TEXT_FIELD = "text"

# useful objects
print("Initializing embedding model...")
embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

print("Connecting to Pinecone...")
index = pinecone.Index(INDEX_NAME)
vectorstore = Pinecone(index, embed, TEXT_FIELD)

print("Initializing chat model...")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                 model_name="gpt-3.5-turbo", temperature=0.0)
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(
    search_kwargs=dict(k=3)), return_source_documents=True)


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding(ENCODING_NAME)

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )

    return len(tokens)


def parse_single_document(path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    loader = PyPDFLoader(path)

    return loader.load_and_split(text_splitter)


def upsert_documents(texts: list, metadatas: list):
    print(f"Upserting {len(texts)} documents.")

    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))


def process_documents_for_upsert(documents: List[Document]):
    texts = []
    metadatas = []

    for _, record in enumerate(tqdm(documents)):
        metadata = {
            'page': str(record.metadata['page']),
            'source': record.metadata['source'],
            'title': record.metadata['source'],
            'text': record.page_content,
        }

        texts.append(record.page_content)
        metadatas.append(metadata)

        if len(texts) >= UPSERT_BATCH_LIMIT:
            upsert_documents(texts, metadatas)

            texts = []
            metadatas = []

    if len(texts) > 0:
        upsert_documents(texts, metadatas)


def retrieve_files_from_folderpath(folderpath: str, extensions: list[str]):
    file_list = []

    for root, _, files in os.walk(folderpath):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))

    return file_list

# TODO: write better crawling logic (including updating based on timestamp and ignoring duplicates)
def crawl_and_upsert():
    print()
    filepaths = []

    while (True):
        crawling_option = input(
            "Would you like to (1) select individual files or (2) select a folder? ")
        if crawling_option == "1":
            filepaths = filedialog.askopenfilenames(
                filetypes=[(f"{ext} files", f"*{ext}") for ext in EXTENSIONS])
            break
        elif crawling_option == "2":
            folderpath = filedialog.askdirectory()
            filepaths = retrieve_files_from_folderpath(folderpath, EXTENSIONS)
            break
        else:
            print("Please enter either 1 or 2")

    if len(filepaths) > 10:
        continue_input = input(
            f"There are {len(filepaths)} files with extensions {EXTENSIONS} in this folder. Are you sure you want to continue? (y/N): ")
        if continue_input != "y":
            print("Aborting!")
            return

    for filepath in filepaths:
        parsed_document = parse_single_document(filepath)
        process_documents_for_upsert(parsed_document)


def print_output(output: Dict[str, Any]):
    print(str(output["answer"]))
    print("Source: " + str(output["sources"]))
    print("Retrieved sources:")
    for source_document in output["source_documents"]:
        source_document: Document = source_document
        print(
            f"- {str(source_document.metadata['source'])} (page {source_document.metadata['page']})")


# TODO: separate query answering and upserting into different files
def answer_queries():
    while (True):
        print()
        query = input("Enter query or type exit: ")
        print("------------")

        if query == "exit":
            break

        output = qa(query)
        print_output(output)


if __name__ == "__main__":
    while (True):
        print()
        choice = input(
            "Would you like to (1) answer queries, (2) add files to the db, (3) clear the current index, or (4) exit? ")
        if choice == "1":
            answer_queries()
        elif choice == "2":
            crawl_and_upsert()
        elif choice == "3":
            confirm = input(
                f"There are {index.describe_index_stats()['total_vector_count']} stored vectors. Are you sure you want to delete? (y/N): ")
            if confirm == "y":
                index.delete(delete_all=True)
                print("Index cleared!")
            else:
                print("Aborting!")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Please enter either 1, 2, 3, or 4")
