import os
import time
import json
import csv
import io
import uuid
from openai import AzureOpenAI
from io import BytesIO
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ----------------------
# 🔧 Setup
# ----------------------
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
BLOB_FILE = os.getenv("AZURE_BLOB_FILE")  # e.g. products.csv

# Initialize Azure OpenAI
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,  
    api_version="2024-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Initialize Azure Search client
search_client = SearchClient(endpoint=SEARCH_ENDPOINT,
                             index_name=SEARCH_INDEX_NAME,
                             credential=AzureKeyCredential(SEARCH_API_KEY))

# Initialize Blob client
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)

# ----------------------
# 📄 Load product docs from Blob
# ----------------------
def load_docs_from_blob(blob_connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    blob_client = blob_service_client.get_blob_client(
        container=container_name, 
        blob=blob_name
    )

    # Download the blob content to a BytesIO object (in-memory)
    # You can also download to a local file using download_to_path()
    download_stream = BytesIO()
    blob_client.download_blob().readinto(download_stream)

    # Reset stream position to the beginning to read content
    download_stream.seek(0)

    # Now you can read the content from download_stream
    blob_content = download_stream.read().decode('utf-8') # Decode if it's text

    docs = []
    if blob_name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(blob_content), skipinitialspace=True)
        for row in reader:
            docs.append(dict(row))

    elif blob_name.endswith(".json"):
        docs = json.loads(blob_content)

    else:
        raise ValueError(f"Unsupported file format: {blob_name}")

    # sanity check (first 3 docs)
    print("✅ Loaded", len(docs), "docs")

    return docs

# ----------------------
# 🧹 Clean + transform fields
# ----------------------
# ----------------------
# 🧹 Transform CSV to index docs
# ----------------------
def transform_doc(raw_doc):
    """
    Converts CSV row to Azure Search document format
    """
    chunk_id = str(uuid.uuid4())
    parent_id = raw_doc.get("parent_id", "") or ""
    title = raw_doc.get("name", "")
    main_category = raw_doc.get("main_category", "")
    sub_category = raw_doc.get("sub_category", "")
    chunk = (
        f"{title}. "
        #f"Category: {raw_doc.get('main_category', '')}/{raw_doc.get('sub_category', '')}. "
        #f"Rating: {raw_doc.get('ratings', '')} from {raw_doc.get('no_of_ratings', '')} users. "
        #f"Price: {raw_doc.get('discount_price', '')} (was {raw_doc.get('actual_price', '')})"
    )

    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "main_category": main_category,
        "sub_category": sub_category,
        "chunk": chunk
    }

# ----------------------
# 🧠 Get embeddings w/ retry
# ----------------------
def get_embedding(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model=AZURE_OPENAI_DEPLOYMENT,  # your deployment name
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limited. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed after retries")

# ----------------------
# 🚀 Process + Upload
# ----------------------
def process_and_upload(batch_size=100):
    raw_docs = load_docs_from_blob(BLOB_CONNECTION_STRING, BLOB_CONTAINER, BLOB_FILE)
    batch = []

    for i, raw_doc in enumerate(raw_docs, start=1):

        doc = transform_doc(raw_doc)

        #print(doc)

        embedding = get_embedding(doc["chunk"])

        # Attach embedding to doc
        doc["text_vector"] = embedding

        batch.append(doc)

        # Upload in batches
        if len(batch) >= batch_size:
            print('uploading...')
            search_client.upload_documents(documents=batch)
            print('done')
            print(f"Uploaded {i} docs")
            batch = []
            time.sleep(1)  # small delay

    # Upload leftover
    if batch:
        search_client.upload_documents(documents=batch)
        print(f"Uploaded remaining {len(batch)} docs")


if __name__ == "__main__":
    process_and_upload(batch_size=50)
