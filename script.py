import os
import tiktoken
import pinecone
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from uuid import uuid4

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

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

    for _, record in enumerate(documents):
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
        upsert_documents


# TODO: write better crawling logic (including updating based on timestamp and ignoring duplicates)
def crawl_and_upsert():
    # folder_path = input()
    file_path = "test-data/pdf/ebay.pdf"


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
    answer_queries()
