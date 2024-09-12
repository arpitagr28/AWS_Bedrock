import os
import uuid
import time
import json
import boto3
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from langchain_aws import BedrockLLM  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    return vectorstore_faiss

def get_response(vectorstore, question):
    prompt_template = """
        Human: Please use the given context to provide a concise answer to the question.
        If you don't know the answer, just say that you don't know; don't try to make up an answer.
        <context>
        {context}
        </context>
        Question: {question}
        Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock_client,
                     model_kwargs={"max_gen_len": 512, "temperature": 0.5, "top_p": 0.9})

    qa = RetrievalQA.from_chain_type(llm=llm,
                                    retriever=vectorstore.as_retriever(
                                        search_kwargs={"k": 5}),
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT})

    answer = qa({"query": question})
    return answer['result']


api = Flask(__name__)
@api.route('/chat', methods=['POST'])

def process_pdf_and_question():
    
    pdf_files = request.files.getlist("pdf_files")
    user_question = request.form.get("question")

    print(">>>>>>>>>>>>>>>>>>>>>>>>>",pdf_files,user_question)
    # Extract file paths from file storage objects
    file_paths = [f.filename for f in pdf_files]

    loader = PyPDFLoader(file_paths)
    pages = loader.load_and_split()
    raw_text = split_text(pages, 1000, 100)
    vector_store = create_vector_store(raw_text)
    response = get_response(vector_store, user_question)

    return jsonify({"response": response["output_text"]})

if __name__ == "__main__":
    api.run(debug=True, port=8585)
