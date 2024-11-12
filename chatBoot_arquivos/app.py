import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings


#####################################################################
## Variaveis de configuração 
## scopo global
#####################################################################

file_directory="arquivos"
embedding_model='sentence-transformers/all-MiniLM-L6-v2'
llm_model ="llama3.2"


#######################################################################
## Carregamento dos dados
## arquivos
## 
#######################################################################
## leitura dos dados em arquivos
## nesse caso pdf e csv
def prepare_and_split_docs(directory):
    # Load the documents
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf",show_progress=True, loader_cls=PyPDFLoader),
        DirectoryLoader(directory, glob="**/*.csv",loader_cls=CSVLoader)
    ]

    ## carrega os dados em uma lista
    documents=[]
    for loader in loaders:
        data =loader.load()
        documents.extend(data)
    

    # inicio da vetorização do texto
    ## estamos colocando as  chunks em um tamanho pequeno
    ## a explicação de onde eu tirei era para evitar 
    # a repetição "spliting logic" agredito que seja trechos repetidos
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,  
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "] ## pode ser usado mais de um separador
    )

    # faz os cortes nos textos conforemindicado
    ## e mantem os metadados juntos
    split_docs = splitter.split_documents(documents)

    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs


########################################################################
##Salvar na base de dados vetorial local
## FAISS (Facebook AI Similarity Search) 
#########################################################################

embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
def ingest_into_vectordb(split_docs):
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Documents are inserted into FAISS vectorstore")
    return db

#######################################################################
#### Vamos começar a conversa
############################################################################


def get_conversation_chain(retriever):
    llm = Ollama(model=llm_model)
    
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided  documents. "
        "Do not rephrase the question or ask follow-up questions."
    )

    ## colocando o prompt, junto com 
    ## o historico do chat e a entrada do humano
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    
    ### prompt para responder nas questões ###
    ### inicio de QA
    system_prompt = (
         "As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-3 sentences. "
        "Answe should be limited to 50 words and 2-3 sentences."
        "and  do not prompt to select answers or do not formualate a stand alone question."
        "do not ask questions in the response. "
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    ## criando a resposta para a questão
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### para gerenciar o  historico do bate papo ###
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print("Conversational chain created")
    return conversational_rag_chain


##############################################################
## Montar o RAG, com os textos
## base de dados 
## fazer o modelo da conversa com retriver
##############################################################
## ler documentos e vetorizar
split_docs=prepare_and_split_docs(file_directory)  
# salvar as chunks na base de dados
vector_db= ingest_into_vectordb(split_docs)  
## criar o canal de comunicação da base vetorial para pesquisa
retriever =vector_db.as_retriever() 
## criar um canal de chat com retriever (utilizar os dados da base)
## para dar contexto
conversational_rag_chain=get_conversation_chain(retriever)

####################################################################
## executar o chatboot
#####################################################################

def generate_response(input):
    qa1=conversational_rag_chain.invoke(
    {"input": input},
    config={
        "configurable": {"session_id": "abc123"}
    }
    )

    return qa1["answer"]

#######################################################################
## Interface grafica 
########################################################################
st.title("🤖 LLMS Chatboot ")
st.write("Desenvolvido para fins educacionais")

with st.form("llm-form"):
    text = st.text_area("Entre com seu texto")
    submit = st.form_submit_button("Enviar")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if submit and text:
    with st.spinner("Gerando a Resposta ... "):
        response = generate_response(text)
        st.session_state['chat_history'].append({"human":text,
                                                 "system":response})
        st.write(response)

#st.write("## Chat History ## ")   
#for chat in st.session_state['chat_history']:
#    st.write(f"**🙂**: {chat['human']}")
#    st.write(f"**🤖 Assitante**: {chat['system']}")
#    st.write("------")