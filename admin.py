import os
import uuid
import boto3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS


folder_path = "/tmp/"
all_files = os.listdir(folder_path)
for file_name in all_files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        os.rmdir(file_path)


## s3
s3 = boto3.client("s3")
s3 = boto3.resource(
                        service_name='s3',
                        region_name='us-east-1',
                        aws_access_key_id='AKIA5FTZEOH72OOLUTPY',
                        aws_secret_access_key='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'
                    )

bedrock_client = boto3.client(service_name="bedrock-runtime")
# BUCKET_NAME = os.getenv("chatbotembedding")

def get_unique_id():
    return str(uuid.uuid4())

# ## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
def create_vector_store(request_id,documents):

    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    # print(vectorstore_faiss)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    
    s3.Bucket("chatbotembedding").upload_file(Filename=folder_path + "/" + file_name + ".faiss" , Key="my_faiss.faiss")
    s3.Bucket("chatbotembedding").upload_file(Filename=folder_path + "/" + file_name + ".pkl" , Key="my_faiss.pkl")

    return True

# ## main method
def main():

    st.write("This is Admin Site for Chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"

        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 100)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id,splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
        else:
            st.write("Error!! Please check logs.")



if __name__ == "__main__":
    main()