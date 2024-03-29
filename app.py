import os
import tempfile
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import shutil
import psutil
import traceback
from tqdm import tqdm
import streamlit as st
from huggingface_hub import login


load_dotenv()
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")

def delete_vector_db_dir(vector_db_dir):
    if os.path.exists(vector_db_dir):
        # Find and terminate processes holding the vector database files
        for proc in tqdm(psutil.process_iter(['name', 'exe', 'open_files'])):
            try:
                for file in proc.info['open_files'] or []:
                    if vector_db_dir in file.path:
                        print(f"Terminating process holding {file.path}: {proc.info['name']} ({proc.pid})")
                        proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Delete the vector database directory
        shutil.rmtree(vector_db_dir, ignore_errors=True)
    os.makedirs(vector_db_dir, exist_ok=True)

def load_documents():
    os.makedirs('data', exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    with st.spinner("Loading Documents..."):
        loader = PyPDFDirectoryLoader(TMP_DIR.as_posix())
        documents = loader.load()
    return documents

def split_documents(documents):
    print("splitting documents")
    with st.spinner("Chunking Documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len, is_separator_regex=False)

    texts = text_splitter.split_documents(documents)
    return texts


def embeddings_on_local_vectordb(texts):
    delete_vector_db_dir(LOCAL_VECTOR_STORE_DIR.as_posix())
    print("embedding started")
    retriever = None
    try:
        with st.spinner(f'{st.session_state.hf_embed_model} is Embedding your PDF(s)...'):
            vectordb = Chroma.from_documents(texts, embedding= HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_name=st.session_state.hf_embed_model),
                                            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix(),
                                            )
            vectordb.persist()
            retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        st.success('I am Ready to Answer your Questions!')
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())  # print full traceback
        
    return retriever

def query_llm(retriever, query):
    print(st.session_state.hf_llm)
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print(f"API token: {api_token}")  # print the API token
    qa_chain = ConversationalRetrievalChain.from_llm(
                    llm= HuggingFaceEndpoint(
                    repo_id=st.session_state.hf_llm, temperature=0.1, token=api_token),
                    retriever=retriever)
    
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def input_fields():
    with st.sidebar:
        st.session_state.hf_llm = st.selectbox('HuggingFace LLM', options=["mistralai/Mistral-7B-Instruct-v0.2", 'google/gemma-7b', 'mistralai/Mixtral-8x7B-Instruct-v0.1', "microsoft/phi-2", "HuggingFaceH4/zephyr-7b-beta"])

        hf_embed_model_prev = st.session_state.get("hf_embed_model", None)
        st.session_state.hf_embed_model = st.selectbox('HuggingFace Embedding', options=["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5", 'jinaai/jina-embeddings-v2-base-en', "intfloat/multilingual-e5-large"], on_change=handle_embed_model_change, kwargs={"prev_model": hf_embed_model_prev})

    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def handle_embed_model_change(prev_model):
    if prev_model != st.session_state.hf_embed_model:
        st.session_state.clear()
        delete_vector_db_dir(LOCAL_VECTOR_STORE_DIR.as_posix())
        st.experimental_rerun()

def process_documents():
    if not st.session_state.hf_llm or not st.session_state.hf_embed_model or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())

            documents = load_documents()

            for _file in TMP_DIR.iterdir():
                temp_file = TMP_DIR.joinpath(_file)
                temp_file.unlink()

            texts = split_documents(documents)
            st.session_state.retriever = embeddings_on_local_vectordb(texts)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    input_fields()
    st.button("Submit Documents", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    

    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()