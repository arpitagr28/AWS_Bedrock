import boto3
import streamlit as st
import os
import uuid
import time
import awscli
#  aws configure  
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


s3 = boto3.client("s3")
s3 = boto3.resource(
                        service_name='s3',
                        region_name='us-east-1',
                        aws_access_key_id='AKIA5FTZEOH72OOLUTPY',
                        aws_secret_access_key='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'
                    )

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():

    s3.Bucket("chatbotembedding").download_file(Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3.Bucket("chatbotembedding").download_file(Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

# get_response()
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
    st.header("This is Client Site for Chat with PDF demo using Bedrock, RAG etc")
    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    # Create FAISS index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")

    if st.button("Ask Question"):
        start_time = time.time()
        with st.spinner("Querying..."):
            st.write(get_response(faiss_index, question))
            st.success("Done")
        end_time = time.time()
        time_taken = end_time - start_time
        st.success(f"Done in {time_taken:.2f} seconds")
        
if __name__ == "__main__":
    main()