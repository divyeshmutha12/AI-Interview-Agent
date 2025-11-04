"""
Document Parser for Resume and Job Description Files

Supports: PDF, DOCX, TXT files
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# PDF parsing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# DOCX parsing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Alternative PDF parser (if PyPDF2 doesn't work well)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parse various document formats (PDF, DOCX, TXT) to extract text content.

    Supports multiple PDF parsing libraries for better compatibility.
    """

    def __init__(self, prefer_pdfplumber: bool = True):
        """
        Initialize the document parser.

        Args:
            prefer_pdfplumber: If True and available, use pdfplumber instead of PyPDF2
                             (pdfplumber often gives better results)
        """
        self.prefer_pdfplumber = prefer_pdfplumber

        # Log available parsers
        parsers = []
        if PDF_AVAILABLE:
            parsers.append("PyPDF2")
        if PDFPLUMBER_AVAILABLE:
            parsers.append("pdfplumber")
        if DOCX_AVAILABLE:
            parsers.append("python-docx")

        if parsers:
            logger.info(f"Document parser initialized with: {', '.join(parsers)}")
        else:
            logger.warning("No document parsing libraries found. Only TXT files will be supported.")

    def parse_file(self, file_path: str) -> str:
        """
        Parse a document file and extract its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            Exception: If parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower()

        # Route to appropriate parser
        if extension == '.pdf':
            return self._parse_pdf(file_path)
        elif extension == '.docx':
            return self._parse_docx(file_path)
        elif extension == '.txt':
            return self._parse_txt(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: .pdf, .docx, .txt"
            )

    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF file using available PDF library"""

        # Try pdfplumber first if preferred and available
        if self.prefer_pdfplumber and PDFPLUMBER_AVAILABLE:
            try:
                return self._parse_pdf_pdfplumber(file_path)
            except Exception as e:
                logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
                if PDF_AVAILABLE:
                    return self._parse_pdf_pypdf2(file_path)
                raise

        # Try PyPDF2
        if PDF_AVAILABLE:
            try:
                return self._parse_pdf_pypdf2(file_path)
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
                if PDFPLUMBER_AVAILABLE:
                    logger.info("Trying pdfplumber as fallback...")
                    return self._parse_pdf_pdfplumber(file_path)
                raise

        # Try pdfplumber as fallback
        if PDFPLUMBER_AVAILABLE:
            return self._parse_pdf_pdfplumber(file_path)

        raise RuntimeError(
            "No PDF parsing library available. "
            "Install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber"
        )

    def _parse_pdf_pypdf2(self, file_path: Path) -> str:
        """Parse PDF using PyPDF2"""
        logger.info(f"Parsing PDF with PyPDF2: {file_path.name}")

        text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            logger.info(f"Processing {num_pages} pages...")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        result = '\n\n'.join(text)
        logger.info(f"Extracted {len(result)} characters from PDF")
        return result

    def _parse_pdf_pdfplumber(self, file_path: Path) -> str:
        """Parse PDF using pdfplumber (often better quality)"""
        logger.info(f"Parsing PDF with pdfplumber: {file_path.name}")

        text = []
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            logger.info(f"Processing {num_pages} pages...")

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        result = '\n\n'.join(text)
        logger.info(f"Extracted {len(result)} characters from PDF")
        return result

    def _parse_docx(self, file_path: Path) -> str:
        """Parse DOCX file"""
        if not DOCX_AVAILABLE:
            raise RuntimeError(
                "python-docx not installed. "
                "Install it: pip install python-docx"
            )

        logger.info(f"Parsing DOCX: {file_path.name}")

        doc = Document(file_path)
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        result = '\n\n'.join(paragraphs)
        logger.info(f"Extracted {len(result)} characters from DOCX")
        return result

    def _parse_txt(self, file_path: Path) -> str:
        """Parse plain text file"""
        logger.info(f"Reading TXT file: {file_path.name}")

        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.info(f"Read {len(content)} characters with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue

        raise RuntimeError(f"Failed to read file with any supported encoding: {encodings}")

    def parse_resume_and_jd(
        self,
        resume_path: str,
        jd_path: str
    ) -> Dict[str, str]:
        """
        Parse both resume and job description files.

        Args:
            resume_path: Path to resume file
            jd_path: Path to job description file

        Returns:
            Dict with 'resume' and 'job_description' keys containing extracted text
        """
        logger.info("Parsing resume and job description...")

        result = {
            'resume': self.parse_file(resume_path),
            'job_description': self.parse_file(jd_path)
        }

        logger.info(
            f"Parsing complete: Resume ({len(result['resume'])} chars), "
            f"JD ({len(result['job_description'])} chars)"
        )

        return result

    @staticmethod
    def get_supported_formats() -> list:
        """Get list of supported file formats"""
        formats = ['.txt']
        if PDF_AVAILABLE or PDFPLUMBER_AVAILABLE:
            formats.append('.pdf')
        if DOCX_AVAILABLE:
            formats.append('.docx')
        return formats

    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Check if file format is supported"""
        extension = Path(file_path).suffix.lower()
        return extension in DocumentParser.get_supported_formats()


# Convenience function for quick usage
def extract_text_from_file(file_path: str) -> str:
    """
    Quick utility to extract text from a file.

    Args:
        file_path: Path to document file

    Returns:
        Extracted text
    """
    parser = DocumentParser()
    return parser.parse_file(file_path)
