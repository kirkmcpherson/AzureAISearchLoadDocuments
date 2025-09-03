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
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

def get_data_from_pdf(pdf_url):
  loader = PyPDFLoader(pdf_url)
  return loader.load()

def split_text_into_chunks(data, chunk_size=400, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  return text_splitter.split_documents(data)


# ----------------------
# 🧹 Clean + transform fields
# ----------------------
# ----------------------
# 🧹 Transform CSV to index docs
# ----------------------
def transform_doc(raw_doc):
    chunk_id = str(uuid.uuid4())
    parent_id = raw_doc.get("parent_id", "") or ""
    title = raw_doc.get("name", "")
    chunk = (
        f"{title}. "
        #f"Category: {raw_doc.get('main_category', '')}/{raw_doc.get('sub_category', '')}. "
        #f"Rating: {raw_doc.get('ratings', '')} from {raw_doc.get('no_of_ratings', '')} users. "
        #f"Price: {raw_doc.get('discount_price', '')} (was {raw_doc.get('actual_price', '')})"
    )

    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
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
    data = get_data_from_pdf("https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf")
    #data = get_data_from_pdf("https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2023.pdf")

    batch = []
    documents = split_text_into_chunks(data, 400, 20)

    for i, raw_doc in enumerate(documents, start=1):

        chunk_id = str(uuid.uuid4())
        title = raw_doc.metadata["source"]
        embedding = get_embedding(raw_doc.page_content)

        doc = {            
            "chunk_id": chunk_id,
            "parent_id": "",
            "title": title,
            "chunk": raw_doc.page_content,
            "text_vector": embedding
        }

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
