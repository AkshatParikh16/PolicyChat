import pdfplumber
from docx import Document
import os
from app.logger.logger import get_logger

logger = get_logger(__name__)

def load_pdf(file_path: str) -> dict:
    """Extract text from a PDF file."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            logger.info(f"PDF loaded: {file_path} | Pages: {len(pdf.pages)}")
            return {
                "text": text.strip(),
                "metadata": {
                    "source": os.path.basename(file_path),
                    "file_type": "pdf",
                    "total_pages": len(pdf.pages)
                }
            }
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise


def load_docx(file_path: str) -> dict:
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        logger.info(f"DOCX loaded: {file_path}")
        return {
            "text": text.strip(),
            "metadata": {
                "source": os.path.basename(file_path),
                "file_type": "docx",
                "total_pages": None
            }
        }
    except Exception as e:
        logger.error(f"Failed to load DOCX {file_path}: {e}")
        raise


def load_txt(file_path: str) -> dict:
    """Extract text from a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        logger.info(f"TXT loaded: {file_path}")
        return {
            "text": text.strip(),
            "metadata": {
                "source": os.path.basename(file_path),
                "file_type": "txt",
                "total_pages": None
            }
        }
    except Exception as e:
        logger.error(f"Failed to load TXT {file_path}: {e}")
        raise


def load_document(file_path: str) -> dict:
    """Auto-detect file type and load document."""
    ext = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt
    }
    
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT")
    
    return loaders[ext](file_path)