# import
import time
from datetime import datetime
import pandas as pd

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


# this will be the prpmpt template for the model

PROMPT_TEMPLATE = """Answer the question as detailed as possible from the provided context. \
Give all the details with proper structure. If the answer is not in the provided context, \
just say "answer is not available in the context". Don't provide a wrong answer.

Context:
{context}

Question:
{question}

Answer:
"""


# getting pdf before dividing into chunks
def get_pdf_texts(pdf_docs):
    text=""
    for pdf in pdf_docs:
        reader= PdfReader(pdf)
        for page in reader.pages:
            text+=(page.extract_text() or "")+ "\n"
    return text

# now lets divide into chunks

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # for debug
    print(splitter.split_text(text))
    return splitter.split_text(text)


# making vector store 

def build_vector_store(chunks , api_key):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key= api_key)
    return FAISS.from_texts(chunks, embedding= embeddings)


def _is_transient(err):
    msg = str(err).upper()
    return any(s in msg for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))


def _is_not_found(err):
    msg = str(err).upper()
    return "404" in msg or "NOT_FOUND" in msg


def answer_question(question, vector_store, api_key, model_names, retries=4):
    """ Try each model in order. 
    - Transient errors (503/429): retry with exponential backoff, then fall through to the next model.
    - Model not found(404): skip straight to the next model.
    - Anything else: Raise
    If every model fails, raise one error that can list what went wrong.
    """
    docs= vector_store.similarity_search(question, k=4)
    context= "\n\n".join(d.page_content for d in docs)
    prompt= PromptTemplate( template= PROMPT_TEMPLATE, input_variables= ['context', 'question'])
    
    errors=[]
    for model_name in model_names:
        llm= ChatGoogleGenerativeAI( model= model_name, temperature= 0.3, google_api_key= api_key)
        chain = prompt | llm | StrOutputParser()
        
        for attempt in range(retries):
            try:
                return chain.invoke({"context": context, "question": question})
            except Exception as e:
                errors.append(f"[{model_name}] {e}")
                if _is_not_found(e):
                    break
                if not _is_transient(e):
                    raise
                time.sleep(2**attempt)
                
                # sleeps for about 1s, 2s, 4s, 8s
                # this may help model to stay overloaded/ not found then can move to next model
    
    #show first nad last errors so a bad fallback also shows real cause 
    summary = errors[0] if len(errors) == 1 else f"{errors[0]}\n...\n{errors[-1]}"
    raise RuntimeError(f"All models failed:\n{summary}")
                

def main():
    st.set_page_config(page_title= "Chat with multple PDFs")
    st.header("Chat with multiple PDFs")
    
    
    # session
    st.session_state.setdefault('history',[])
    
    st.session_state.setdefault("vector_store",None)
    st.session_state.setdefault("pdf_names",[])
    
    with st.sidebar:
        st.title("Menu")
        
        api_key = st.text_input("Google API Key", type= "password")
        st.markdown("Get a key [here](https://aistudio.google.com/app/apikey).")
        st.markdown(
            "Model names change often - check the "
            "[models page](https://ai.google.dev/gemini-api/docs/models)."
        )
        model_name = st.text_input("Chat model", value= 'gemini-3.8-flash')
        fallback_model = st.text_input("Fallback model (used if the first is overloaded, optional )", value= 'gemini-3.5-flash-lite',)
        pdf_docs= st.file_uploader("Upload your PDF files, then click Submit & Process", type=["pdf"], accept_multiple_files=True,)
        if st.button("Submit"):
            if not api_key:
                st.warning("Please enter Google API key first")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                try:
                    with st.spinner("Processing..."):
                        text= get_pdf_texts(pdf_docs)
                        if not text.strip():
                            st.error("No text could be extracted")
                        else:
                            chunks= get_text_chunks(text)
                            st.session_state.vector_store= build_vector_store(chunks, api_key)
                            st.session_state.pdf_names= [p.name for p in pdf_docs]
                            st.session_state.history=[]
                            st.success(f"Done- {len(chunks)} chunks indexed")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
        if st.button("Reset"):
            st.session_state.history=[]
            st.session_state.vector_store=None
            st.session_state.pdf_names =[]
            st.rerun()
        
        if st.session_state.history:
            df= pd.DataFrame(
                [(h["question"], h["answer"], h["time"], ",".join(h["pdfs"])) for h in st.session_state.history],
                columns=["Question", "Answer", "Timestamp", "PDF Names"],
            )
            st.download_button("Download conversation as CSV", df.to_csv(index=False).encode('utf-8'),
            file_name="conversation_history.csv",
            mime="text/csv",)
    
    for h in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(h["question"])
        with st.chat_message("assistant"):
            st.markdown(h["answer"])
    
    question= st.chat_input("Ask a question from pdf files")
    if question:
        if not api_key:
            st.warning("please enter your API key.")
            return 
        if st.session_state.vector_store is None:
            st.warning("Please uplaod PDFs and click Submit first.")
            return 
        
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    models = [
                        m.strip() for m in (model_name, fallback_model)
                        if m.strip()
                    ]
                    answer= answer_question(question, st.session_state.vector_store, api_key, models)
                
                st.markdown(answer)
                st.session_state.history.append(
                    {"question": question, 
                     "answer": answer,
                     "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "pdfs": st.session_state.pdf_names,
                     }
                )
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()           