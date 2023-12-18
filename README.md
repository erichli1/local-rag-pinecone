Basic Python terminal app that allows user to embed and upsert documents into Pinecone and then use RAG to answer questions.

To prevent duplicate uploads, previously-uploaded filepaths are stored locally in `sources.txt` and ignored if tried to be uploaded again.

# Setup

Add `PINECONE_API_KEY`, `INDEX_NAME`, `PINECONE_ENVIRONMENT`, and `OPENAI_API_KEY` to environment.

Run `python3 script.py`