# RAG-powered chat for local files

Basic Python terminal app that allows user to embed and upsert local files into Pinecone and then use RAG to answer questions.

To prevent duplicate uploads, previously-uploaded filepaths are stored locally in `sources.txt` and ignored if tried to be uploaded again.

## Setup

Add `PINECONE_API_KEY`, `INDEX_NAME`, `PINECONE_ENVIRONMENT`, and `OPENAI_API_KEY` to environment.

Install requirements with `pip install -r requirements.txt`. Note that you will have to install the punkt tokenizer and averaged_perceptron_tagger POS tagger from NTLK separately with `import ntlk; ntlk.download('punkt'); ntlk.download('averaged_perceptron_tagger')`.

Run `python3 main.py`