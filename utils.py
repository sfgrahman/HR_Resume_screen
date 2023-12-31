import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.huggingface_hub import HuggingFaceHub
from config import *


def pinecone_settings():
    pinecone.init(
    api_key=api_key,
    environment=environment
    )
    index = index_name
    return index

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        #print(filename)
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))
    return docs


def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def push_to_pinecone(embeddings,docs):
    index_name= pinecone_settings()
    Pinecone.from_documents(docs, embeddings, index_name=index_name)


def pull_from_pinecone(index_name, embeddings):
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,embeddings,unique_id):
    index_name = pinecone_settings()
    index = pull_from_pinecone(index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary


