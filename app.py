import json
import os
import sys
import boto3
import streamlit as st

## Titan Embeddings Model
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

# bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents=loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings,
    )
    vectorstore_faiss.save_local("faiss_index")
 # LLM   
def get_titan_llm():
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        )
    return llm

# Prompt
prompt_template="""
Human: Use the follwoing pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with
250 words with detailed explanations. If you dont know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question:{question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

# response
def get_response_llm(llm,vectorsore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorsore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":2}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    
    answer = qa({"query":query})
    return answer['result']


def main():
    st.set_page_config(
    page_title="AWS Bedrock PDF QA",
    page_icon="🤖",
    layout="wide"
    )
    st.header("Chat with PDFs using AWS Bedrock 🤖")
    
    user_question = st.text_input("Ask a Question from the PDF files")
    output_container = st.container()
    
    with st.sidebar:
        st.title("Menu:")
        
        if st.button("Update or Create Vector Store"):
            with st.spinner("Proccessing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("✨ Get Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
                )
            llm = get_titan_llm()
            streamlit_callback = StreamlitCallbackHandler(output_container)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
    
   #if st.button("🦙 Get Llama Output"):
    #    with st.spinner("Processing..."):
     #       faiss_index = FAISS.load_local(
      #          "faiss_index",
       #         bedrock_embeddings,
        #        allow_dangerous_deserialization=True
         #       )
           # llm = get_llama_llm()
            #st.write(get_response_llm(llm,faiss_index,user_question))
    
if __name__ == "__main__":
    main()
    