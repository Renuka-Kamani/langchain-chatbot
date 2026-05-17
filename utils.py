from pypdf import PdfReader
from docx import Document

def extract_text(file):

    text = ""

    file_name = file.name.lower()

    # PDF
    if file_name.endswith(".pdf"):

        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    # TXT
    elif file_name.endswith(".txt"):

        text = str(file.read(), "utf-8")

    # DOCX
    elif file_name.endswith(".docx"):

        doc = Document(file)

        for para in doc.paragraphs:
            text += para.text + "\n"

    return text