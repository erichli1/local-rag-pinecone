import os
import tiktoken
from tqdm.auto import tqdm
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from uuid import uuid4
from tkinter import filedialog

from constants import ENCODING_NAME, EXTENSIONS, LOCAL_SOURCES_FILEPATH, MAX_CHUNK_SIZE, UPSERT_BATCH_LIMIT
from setup import setup

existing_sources = []


def update_existing_sources():
    global existing_sources

    existing_sources = []
    with open(LOCAL_SOURCES_FILEPATH, 'r') as file:
        for line in file:
            existing_sources.append(line.strip())


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


def upsert_documents(embed: OpenAIEmbeddings, texts: list, metadatas: list):
    print(f"Upserting {len(texts)} documents.")

    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))


def get_title_from_filepath(filepath: str):
    return os.path.basename(filepath)


def process_documents_for_upsert(embed: OpenAIEmbeddings, documents: List[Document]):
    texts = []
    metadatas = []

    for _, record in enumerate(tqdm(documents)):
        metadata = {
            'page': str(record.metadata['page']),
            'source': record.metadata['source'],
            'title': get_title_from_filepath(record.metadata['source']),
            'text': record.page_content,
        }

        texts.append(record.page_content)
        metadatas.append(metadata)

        if len(texts) >= UPSERT_BATCH_LIMIT:
            upsert_documents(embed, texts, metadatas)

            texts = []
            metadatas = []

    if len(texts) > 0:
        upsert_documents(embed, texts, metadatas)


def retrieve_files_from_folderpath(folderpath: str, extensions: list[str]):
    file_list = []

    for root, _, files in os.walk(folderpath):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))

    return file_list


def crawl_and_upsert(embed: OpenAIEmbeddings, previously_upserted: bool):
    global existing_sources

    print()
    filepaths = []

    # NOTE: on older versions of tkinter, repeated calls will result in segfault.
    # https://github.com/python/cpython/issues/92603
    while (True):
        if previously_upserted:
            print(
                "Note that older versions of tkinter will error if you open up the select dialog multiple times. ", end="")
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

    previously_upserted = True

    if len(filepaths) > 10:
        continue_input = input(
            f"There are {len(filepaths)} files with extensions {EXTENSIONS} in this folder. Are you sure you want to continue? (y/N): ")
        if continue_input != "y":
            print("Aborting!")
            return

    upserted_filepaths = []

    for filepath in filepaths:
        # TODO: account for updated files
        if filepath in existing_sources:
            print(
                f"Skipping {get_title_from_filepath(filepath)} because it was already parsed.")
        else:
            # TODO: only call this once on entire set of documents
            parsed_document = parse_single_document(filepath)
            process_documents_for_upsert(embed, parsed_document)
            upserted_filepaths.append(filepath)

    with open(LOCAL_SOURCES_FILEPATH, 'a') as file:
        for filepath in upserted_filepaths:
            file.write(filepath + "\n")

    update_existing_sources()


def print_output(output: Dict[str, Any]):
    print(str(output["answer"]))
    print("Source: " + str(output["sources"]))
    print("Retrieved sources:")
    for source_document in output["source_documents"]:
        source_document: Document = source_document
        print(
            f"- {str(source_document.metadata['title'])} (page {source_document.metadata['page']})")


# TODO: separate query answering and upserting into different files
def answer_queries(qa: BaseQAWithSourcesChain):
    while (True):
        print()
        query = input("Enter query or type exit: ")
        print("------------")

        if query == "exit":
            break

        output = qa(query)
        print_output(output)


if __name__ == "__main__":
    embed, index, vectorstore, qa = setup()

    previously_upserted = False
    update_existing_sources()

    while (True):
        print()
        choice = input(
            "Would you like to (1) answer queries, (2) add files to the db, (3) clear the current index, or (4) exit? ")
        if choice == "1":
            answer_queries(qa)
        elif choice == "2":
            crawl_and_upsert(embed, previously_upserted)
        elif choice == "3":
            confirm = input(
                f"There are {index.describe_index_stats()['total_vector_count']} stored vectors. Are you sure you want to delete? (y/N): ")
            if confirm == "y":
                index.delete(delete_all=True)
                with open(LOCAL_SOURCES_FILEPATH, 'w') as file:
                    pass
                update_existing_sources()
                print("Index cleared!")
            else:
                print("Aborting!")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Please enter either 1, 2, 3, or 4")
