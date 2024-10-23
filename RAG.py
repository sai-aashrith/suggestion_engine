import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os
from typing import List
from chromadb.api.types import EmbeddingFunction
import shutil

load_dotenv()

# Set up the OpenAI embedding model
embed_model = OpenAIEmbedding()

# Set up the OpenAI language model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

Settings.llm = llm
Settings.embed_model = embed_model

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.embed_model.get_text_embedding_batch(input)
        return embeddings

# Remove existing ChromaDB directory if it exists
chroma_db_path = "./chroma_db"
if os.path.exists(chroma_db_path):
    shutil.rmtree(chroma_db_path)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=chroma_db_path)

# Create a new collection with the correct configuration
collection = client.create_collection(
    name="personal_info",
    embedding_function=OpenAIEmbeddingFunction(embed_model)
)

# Sample data insertion
sample_data = [
    {
        "first_name": "John",
        "email": "john.doe@example.com",
        "phone_number": "+1234567890",
        "date_of_birth": "1985-03-15",
        "gender": "Male",
        "address": "123 Main St",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "zip_code": "10001"
    },
    {
        "first_name": "Jane",
        "email": "jane.smith@example.com",
        "phone_number": "+0987654321",
        "date_of_birth": "1990-07-22",
        "gender": "Female",
        "address": "456 Elm St",
        "city": "Los Angeles",
        "state": "CA",
        "country": "USA",
        "zip_code": "90001"
    },
    {
        "first_name": "Alice",
        "email": "alice.johnson@example.com",
        "phone_number": "+1122334455",
        "date_of_birth": "1988-11-10",
        "gender": "Female",
        "address": "789 Oak St",
        "city": "Chicago",
        "state": "IL",
        "country": "USA",
        "zip_code": "60601"
    }
]

# Insert sample data into ChromaDB
for i, data in enumerate(sample_data):
    collection.add(
        documents=[str(data)],
        metadatas=[data],
        ids=[f"doc_{i}"]
    )

# Create a ChromaVectorStore instance
vector_store = ChromaVectorStore(chroma_collection=collection)

# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create a list of documents for LlamaIndex
documents = [Document(text=str(data), id_=f"doc_{i}") for i, data in enumerate(sample_data)]

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# Define the prompt template
prompt_template = """
You are a recommendation engine for form fields. 
Based on the user's query, suggest the next field that the user might want to add.
The next field you suggest be should be a logical field that follows the current field the user is interested in.
Do not suggest the same field that is already mentioned in the query.
Make sure that the field you suggest is a field in the db.

User query:
{query}

"""


def suggest_follow_up_field(query):
    # Format the prompt with the query
    prompt = prompt_template.format(query=query)

    # Query the index to get a response
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)

    # Extract the relevant part of the response
    suggestion = response.response

    return suggestion

query = "I want to create a field 'first name'."
response = suggest_follow_up_field(query)
print(response)

# Example usage
query = "I would like to add a field for 'height'."
response = suggest_follow_up_field(query)
print(response)