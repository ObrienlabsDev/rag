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


#getpass.getpass()

llm = ChatOpenAI(model="gpt-4o-mini")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["OPENAI_API_KEY"] = "sk-p..."
#os.environ["LANGCHAIN_API_KEY"] = "lsv..."
#getpass.getpass()

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

print(f"docs content size: %s" % (len(docs[0].page_content)))
print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000, chunk_overlap=200) #  add_start_index=True
splits = text_splitter.split_documents(docs)
print(f"splits: %s" % (len(splits)))
print(f"page_content: %s" % (len(splits[0].page_content)))
print(f"metadata: %s" % (len(splits[10].metadata)))

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

  # Retrieve and generate using the relevant snippets of the blog.
#retriever = vectorstore.as_retriever() # default simularity, 4
retriever = vectorstore.as_retriever(search_type="similarity", 
                                     search_kwargs={"k": 6})

print (f"retriever: %s" % (retriever))
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
print(f"vectorstore retrieved: %s" % (len(retrieved_docs)))
print(f"vectorstore retrieved content: %s" % (retrieved_docs[0].page_content))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
#example_messages
print(f"example_messages: %s" % (example_messages[0].content))

rag_chain = (
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
  )

#for chunk in rag_chain.stream("What is Task Decomposition?"):
#    print(chunk, end="", flush=True)

retrieved = rag_chain.invoke("What is Task Decomposition?")
print(f"rag_chain retrieved: %s" % (len(retrieved)))
print(f"rag_chain retrieved content: %s" % (retrieved))#[0].page_content))