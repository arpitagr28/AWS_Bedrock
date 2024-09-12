import os
import uuid
import time
import boto3
import streamlit as st
from flask import Flask
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

# ## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    return vectorstore_faiss

def get_response(vectorstore,question):

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
        input_variables=["context", "question"])

    llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock_client,
                     model_kwargs={"max_gen_len": 512, "temperature": 0.5, "top_p": 0.9})

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     retriever=vectorstore.as_retriever(
                                     search_kwargs={"k": 5}),
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT})

    answer = qa({"query": question})
    return answer['result']

# Main function
def main():

    st.header("This is Admin Site for Chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    
    if uploaded_file is not None:

        request_id = get_unique_id()
        # st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"

        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        ## Split Text
        splitted_docs = split_text(pages, 1000, 100)
        st.write(f"Total Pages: {len(pages)}")
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        vectorstore = create_vector_store(splitted_docs)

        if vectorstore:
            st.write("Hurray!! PDF processed and vectorbase is created successfully")
        else:
            st.write("Error!! Please check logs.")


        st.write(f"The model is getting ready")
        question = st.text_input("Please ask your question")

        if st.button("Ask Question"):
            with st.spinner("Querying..."):
                start_time = time.time()
                st.write(get_response(vectorstore, question))
                end_time = time.time()
                st.success("Done")
            time_taken = end_time - start_time
            st.success(f"Done in {time_taken:.2f} seconds")


if __name__ == "__main__":
    main()