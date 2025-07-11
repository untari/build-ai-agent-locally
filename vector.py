from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# read the cvs file
df=pd.read_csv("realistic_restaurant_reviews.csv")
# define the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# check if this location already exists
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# if not prepare all the data by converting the documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# initialize the vectore store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory= db_location,
    embedding_function= embeddings
)

# if the directory already exists there's no need to add the data.
# but if not add this data into the vector by adding all of documents
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# allowed to grab relevant documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)