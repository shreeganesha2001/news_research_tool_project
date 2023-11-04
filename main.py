import os
import pickle
import time
from langchain.llms import GooglePalm
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
import langchain
from langchain.prompts.prompt import PromptTemplate
import streamlit as st


api_key = "AIzaSyDk1iOI3Z0SqCok_10hURmysu0qR3BnZWo"
llm = GooglePalm(google_api_key=api_key, temperature=0.6)

st.title("Fast Reader: QA Tool")
st.sidebar.title("Any Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",", " ", "|"],
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
    )
    main_placeholder.text("Text Splitter...Started...✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    # create embeddings and save it to FAISS index
    embeddings = GooglePalmEmbeddings(google_api_key=api_key)
    # embeddings=HuggingFaceInstructEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅")
    prompt = PromptTemplate(
        input_variables=["Question", "Text", "sources"],
        template="""Given the following extracted parts of a long document and a question, create a final answer  with references {sources}. \nIf you don\'t know the answer, just say that you don\'t know. Don\'t try to make up an answer.\n
    ALWAYS return a SOURCES part in your answer.\ndocument: {Text} \nQUESTION: {Question}""",
    )
    final_prompt = PromptTemplate(
        input_variables=["Question", "Text"],
        template="""Given the following extracted parts of a long document and a question, create a explain 2 line answer  with references sources. Only return the source which is under the document you used to get output. \nIf you don\'t know the answer, just say that you don\'t know. Don\'t try to make up an answer.\n
        ALWAYS return a SOURCES part in your answer.\ndocument: {Text} \nQUESTION: {Question}""",
    )
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = LLMChain(llm=llm, prompt=prompt)
            final_chain = LLMChain(llm=llm, prompt=final_prompt)
            info = vectorstore.similarity_search(query, k=4)
            sum = ""
            for i in range(4):
                result = chain.run(
                    Question=query,
                    Text=info[i].page_content,
                    sources=info[i].metadata["source"],
                )
                sum = sum + " " + result
            final = final_chain.run(Question=query, Text=sum)
            # result will be a dictionary of this format --> {"answer": ""mlitpip install -U -q google-generativeai, "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            if final:
                st.subheader("Sources:")
                st.write(final)
