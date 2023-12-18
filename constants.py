import os

# TODO: support more file types
EXTENSIONS = [".pdf"]
LOCAL_SOURCES_FILEPATH = "./sources.txt"

# langchain
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
ENCODING_NAME = "cl100k_base"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# pinecone
INDEX_NAME = os.environ["INDEX_NAME"]
MAX_CHUNK_SIZE = 400
UPSERT_BATCH_LIMIT = 100
TEXT_FIELD = "text"
