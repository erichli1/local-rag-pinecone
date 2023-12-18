from typing import Any, Dict
from langchain_core.documents import Document
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain


def get_output_str(output: Dict[str, Any]):
    """Returns a well-formatted string represenation of the output."""
    output_str = str(output["answer"]) + "\n"
    output_str += "Source: " + str(output["sources"]) + "\n"
    output_str += "Retrieved sources:\n"

    for source_document in output["source_documents"]:
        source_document: Document = source_document
        output_str += f"- {str(source_document.metadata['title'])} (page {source_document.metadata['page']})\n"

    return output_str


def answer_queries(qa: BaseQAWithSourcesChain):
    """Answers queries from the user with RAG."""
    while (True):
        print()
        query = input("Enter query or type exit: ")
        print("------------")

        if query == "exit":
            break

        output = qa(query)
        print(get_output_str(output))
