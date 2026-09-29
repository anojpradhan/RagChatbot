# import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


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