# from https://python.langchain.com/v0.2/docs/tutorials/rag/
import getpass
import os
import bs4

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
# Configure logging
logging.basicConfig(level=logging.INFO)

def set_environment_variables():
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["LANGCHAIN_API_KEY"] = ""

def initialize_llm():
    return ChatOpenAI(model="gpt-4o-mini")

def load_documents():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

def create_vectorstore(splits):
    return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

def retrieve_documents(vectorstore, query, k=6):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(query)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    set_environment_variables()
    llm = initialize_llm()

    try:
        docs = load_documents()
        logging.info(f"Docs content size: {len(docs[0].page_content)}")
        logging.info(docs[0].page_content[:500])

        splits = split_documents(docs)
        logging.info(f"Splits: {len(splits)}")
        logging.info(f"Page content size: {len(splits[0].page_content)}")
        logging.info(f"Metadata size: {len(splits[10].metadata)}")

        vectorstore = create_vectorstore(splits)
        retrieved_docs = retrieve_documents(vectorstore, "What are the approaches to Task Decomposition?")
        logging.info(f"Vectorstore retrieved: {len(retrieved_docs)}")

        prompt = hub.pull("rlm/rag-prompt")
        example_messages = prompt.invoke(
            {"context": "filler context", "question": "filler question"}
        ).to_messages()
        logging.info(f"Example messages: {example_messages[0].content}")

        # retriever needs to be moved up from function scope
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Uncomment to use the RAG chain
        # for chunk in rag_chain.stream("What is Task Decomposition?"):
        #     logging.info(chunk)
        retrieved = rag_chain.invoke("What is Task Decomposition?")
        logging.info(f"rag_chain retrieved: {len(retrieved)}")
        logging.info(f"rag_chain retrieved content: {retrieved}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()