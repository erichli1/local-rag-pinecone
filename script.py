from typing import Any, Dict
from langchain_core.documents import Document
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain

from constants import LOCAL_SOURCES_FILEPATH
from setup import setup
from update_pinecone import crawl_and_upsert
from utils import update_existing_sources


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
    existing_sources = []

    update_existing_sources(existing_sources)

    while (True):
        print()
        choice = input(
            "Would you like to (1) answer queries, (2) add files to the db, (3) clear the current index, or (4) exit? ")
        if choice == "1":
            answer_queries(qa)
        elif choice == "2":
            crawl_and_upsert(
                embed, index, previously_upserted, existing_sources)
        elif choice == "3":
            confirm = input(
                f"There are {index.describe_index_stats()['total_vector_count']} stored vectors. Are you sure you want to delete? (y/N): ")
            if confirm == "y":
                index.delete(delete_all=True)
                with open(LOCAL_SOURCES_FILEPATH, 'w') as file:
                    pass
                update_existing_sources(existing_sources)
                print("Index cleared!")
            else:
                print("Aborting!")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Please enter either 1, 2, 3, or 4")
