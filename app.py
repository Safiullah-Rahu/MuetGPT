import streamlit as st
import os
import time
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA


st.set_page_config(
    page_title="MUET GPT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("muet.png", width=190)

# Welcome to muetGPT, your virtual assistant for all things related to Mehran University of Engineering and Technology (MUET), Jamshoro. Powered by OpenAI's cutting-edge technology, muetGPT is here to answer all your queries about MUET.

# At MUET, we understand that you may have questions about various aspects of university life, courses, programs, events, facilities, and more. With muetGPT, you can ask any question you have, and our AI-powered assistant will provide you with accurate and helpful information.

# Whether you're a student, faculty member, prospective student, or anyone curious about MUET, muetGPT is here to assist you. Our goal is to make it easier for you to find the information you need and navigate the MUET experience seamlessly.

# Feel free to start asking your questions, and let muetGPT provide you with prompt and reliable answers. Welcome to the future of information access at MUET with muetGPT!
with st.sidebar:
    st.markdown("# Welcome to MuetGPT")
    st.markdown(
        "Your virtual assistant for all things related to Mehran University of Engineering and Technology (MUET), Jamshoro. Powered by OpenAI's cutting-edge technology, muetGPT is here to answer all your queries about MUET."
        )
    st.markdown(
        "At MUET, we understand that you may have questions about various aspects of university life, courses, programs, events, facilities, and more. With muetGPT, you can ask any question you have, and our AI-powered assistant will provide you with accurate and helpful information.👩‍🏫 \n"
    )
    st.markdown("---")
    st.markdown("A project by Safiullah Rahu & Munsif Raza")
    st.markdown("""[![Follow](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/safiullahrahu/)
                   [![Follow](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/munsifraza/)""")
    st.markdown("""[![Follow](https://img.shields.io/twitter/follow/safiullah_rahu?style=social)](https://www.twitter.com/safiullah_rahu)""")
  
# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV
# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


embeddings = OpenAIEmbeddings()
model_name = "gpt-3.5-turbo-16k"
text_field = "text"

@st.cache_resource
def ret():
    # load a Pinecone index
    index = pinecone.Index("muetgpt")
    time.sleep(5)
    db = Pinecone(index, embeddings.embed_query, text_field)
    return db

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
                                        k=2, 
                                        memory_key="chat_history", 
                                        return_messages=True,
                                        verbose=True)

memory = init_memory()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are helpful information giving QA System and make sure you don't answer anything not related to following context. 
You are always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
Also check chat history if question can be answered from it or question asked about previous history. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}
Chat History: {chat_history}
Question: {question}
Answer:"""

qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history","question"]
)

db = ret()

#@st.cache_resource
def conversational_chat(query):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    
    # llm = ChatOpenAI(model=model_name)
    # docs = db.similarity_search(query)
    # qa = load_qa_chain(llm=llm, chain_type="stuff")
    # # Run the query through the RetrievalQA model
    # result = qa.run(input_documents=docs, question=query) #chain({"question": query, "chat_history": st.session_state['history']})
    #st.session_state['history'].append((query, result))#["answer"]))

    # return qa_chain #result   #["answer"]
    llm = ChatOpenAI(model_name = model_name)
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory, verbose=True)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
    agent = ConversationalRetrievalChain(
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory,
        verbose=True,
        # return_source_documents=True,
        # get_chat_history=lambda h :h
    )

    return agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content":prompt})
    # st.chat_message("user").write(prompt)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        agent = conversational_chat(prompt)
        with st.spinner("Thinking..."):
            with get_openai_callback() as cb:
                response = agent({'question': prompt, 'chat_history': st.session_state.chat_history})#agent({"query": prompt})#conversational_chat(prompt)#
                st.session_state.chat_history.append((prompt, response["answer"]))
                #st.write(response)
                message_placeholder.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            #st.sidebar.header(f"Total Token Usage: {cb.total_tokens}")
