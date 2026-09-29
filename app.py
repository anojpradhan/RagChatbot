# import
import time


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
def get_pdf_texts(pdf_doc):
    text=""
    for pdf in pdf_doc:
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
    embeddings= GoogleGenerativeAIEmbeddings(model="modles/gemini-embedding-001", google_api_key= api_key)
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
    prompt= PROMPT_TEMPLATE( template= PROMPT_TEMPLATE, input_variables= ['context', 'question'])
    
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
                