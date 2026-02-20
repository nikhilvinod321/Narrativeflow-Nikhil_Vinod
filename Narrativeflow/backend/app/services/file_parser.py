"""
File parser service for importing existing manuscripts.
Supports DOCX, PDF, TXT, EPUB, RTF formats.
"""

import re
import io
from typing import List, Dict, Optional, BinaryIO
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import chardet
from docx import Document
from PyPDF2 import PdfReader
from striprtf.striprtf import rtf_to_text


class FileParser:
    """Parses various file formats and extracts story content."""
    
    SUPPORTED_FORMATS = ['docx', 'pdf', 'txt', 'epub', 'rtf']
    
    # Common chapter indicators
    CHAPTER_PATTERNS = [
        r'^chapter\s+\d+',
        r'^ch\.\s*\d+',
        r'^\d+\.\s',
        r'^part\s+\d+',
        r'^section\s+\d+',
        r'^# ',  # Markdown heading
        r'^\* \* \*$',  # Scene break
        r'^---$',  # Scene break
    ]
    
    @classmethod
    def detect_format(cls, filename: str) -> Optional[str]:
        """Detect file format from filename."""
        ext = filename.lower().split('.')[-1]
        return ext if ext in cls.SUPPORTED_FORMATS else None
    
    @classmethod
    def parse_file(cls, file_content: bytes, filename: str) -> Dict:
        """
        Parse file and return structured content.
        
        Returns:
            {
                'title': str,
                'content': str,  # Full text content
                'chapters': [{'title': str, 'content': str}, ...],
                'metadata': {}
            }
        """
        file_format = cls.detect_format(filename)
        
        if not file_format:
            raise ValueError(f"Unsupported file format: {filename}")
        
        # Parse based on format
        if file_format == 'docx':
            return cls._parse_docx(file_content, filename)
        elif file_format == 'pdf':
            return cls._parse_pdf(file_content, filename)
        elif file_format == 'txt':
            return cls._parse_txt(file_content, filename)
        elif file_format == 'epub':
            return cls._parse_epub(file_content, filename)
        elif file_format == 'rtf':
            return cls._parse_rtf(file_content, filename)
        
        raise ValueError(f"Parser not implemented for: {file_format}")
    
    @classmethod
    def _parse_docx(cls, file_content: bytes, filename: str) -> Dict:
        """Parse Microsoft Word document."""
        try:
            doc = Document(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Cannot read as DOCX file. Please ensure it's a valid Word 2007+ (.docx) file. Error: {str(e)}")
        
        # Extract metadata
        try:
            core_props = doc.core_properties
            title = core_props.title or filename.rsplit('.', 1)[0]
            author = core_props.author or ''
        except:
            title = filename.rsplit('.', 1)[0]
            author = ''
        
        # Extract text with paragraph structure - be very forgiving
        paragraphs = []
        for para in doc.paragraphs:
            try:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            except Exception as e:
                # Skip problem paragraphs
                print(f"Warning: Skipped paragraph due to error: {e}")
                continue
        
        full_text = '\n\n'.join(paragraphs)
        
        # If no text extracted, this might be an old .doc file or corrupted
        if not full_text.strip():
            raise ValueError("No text could be extracted from the file. This may be an old .doc file (not .docx), empty document, or the file may be corrupted.")
        
        # Detect chapters
        chapters = cls._detect_chapters(paragraphs)
        
        return {
            'title': title,
            'content': full_text,
            'chapters': chapters if chapters else [{'title': 'Chapter 1', 'content': full_text}],
            'metadata': {
                'author': author,
                'format': 'docx',
                'paragraph_count': len(paragraphs)
            }
        }
    
    @classmethod
    def _parse_pdf(cls, file_content: bytes, filename: str) -> Dict:
        """Parse PDF document."""
        try:
            pdf = PdfReader(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {str(e)}")
        
        # Extract metadata
        info = pdf.metadata
        title = info.get('/Title', filename.rsplit('.', 1)[0]) if info else filename.rsplit('.', 1)[0]
        
        # Extract text from all pages
        paragraphs = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Split by double newlines to approximate paragraphs
                page_paras = [p.strip() for p in text.split('\n\n') if p.strip()]
                paragraphs.extend(page_paras)
        
        full_text = '\n\n'.join(paragraphs)
        
        # Detect chapters
        chapters = cls._detect_chapters(paragraphs)
        
        return {
            'title': title,
            'content': full_text,
            'chapters': chapters,
            'metadata': {
                'author': info.get('/Author', '') if info else '',
                'format': 'pdf',
                'page_count': len(pdf.pages),
                'paragraph_count': len(paragraphs)
            }
        }
    
    @classmethod
    def _parse_txt(cls, file_content: bytes, filename: str) -> Dict:
        """Parse plain text file."""
        # Detect encoding - try multiple methods
        detected = chardet.detect(file_content)
        encoding = detected['encoding'] or 'utf-8'
        
        # Decode text with fallback chain for non-English text
        text = None
        encodings_to_try = [encoding, 'utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                text = file_content.decode(enc)
                break
            except (UnicodeDecodeError, AttributeError):
                continue
        
        if text is None:
            # Last resort: decode with errors='replace' to show � for undecodable chars
            text = file_content.decode('utf-8', errors='replace')
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Detect chapters
        chapters = cls._detect_chapters(paragraphs)
        
        return {
            'title': filename.rsplit('.', 1)[0],
            'content': text,
            'chapters': chapters,
            'metadata': {
                'format': 'txt',
                'encoding': encoding,
                'paragraph_count': len(paragraphs)
            }
        }
    
    @classmethod
    def _parse_epub(cls, file_content: bytes, filename: str) -> Dict:
        """Parse EPUB ebook."""
        try:
            book = epub.read_epub(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Failed to parse EPUB file: {str(e)}")
        
        # Extract metadata
        title = book.get_metadata('DC', 'title')
        title = title[0][0] if title else filename.rsplit('.', 1)[0]
        
        author = book.get_metadata('DC', 'creator')
        author = author[0][0] if author else ''
        
        # Extract text from document items
        chapters = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator='\n\n', strip=True)
                
                if text:
                    # Try to extract chapter title from first heading
                    heading = soup.find(['h1', 'h2', 'h3'])
                    chapter_title = heading.get_text(strip=True) if heading else f"Chapter {len(chapters) + 1}"
                    
                    chapters.append({
                        'title': chapter_title,
                        'content': text
                    })
        
        # Combine all chapters for full content
        full_text = '\n\n---\n\n'.join([ch['content'] for ch in chapters])
        
        return {
            'title': title,
            'content': full_text,
            'chapters': chapters if chapters else cls._detect_chapters(full_text.split('\n\n')),
            'metadata': {
                'author': author,
                'format': 'epub',
                'chapter_count': len(chapters)
            }
        }
    
    @classmethod
    def _parse_rtf(cls, file_content: bytes, filename: str) -> Dict:
        """Parse RTF document."""
        # Decode RTF - try multiple encodings
        rtf_text = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                rtf_text = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if rtf_text is None:
            rtf_text = file_content.decode('utf-8', errors='replace')
        
        try:
            plain_text = rtf_to_text(rtf_text)
        except Exception as e:
            raise ValueError(f"Failed to parse RTF content: {str(e)}")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in plain_text.split('\n\n') if p.strip()]
        
        # Detect chapters
        chapters = cls._detect_chapters(paragraphs)
        
        return {
            'title': filename.rsplit('.', 1)[0],
            'content': plain_text,
            'chapters': chapters,
            'metadata': {
                'format': 'rtf',
                'paragraph_count': len(paragraphs)
            }
        }
    
    @classmethod
    def _detect_chapters(cls, paragraphs: List[str]) -> List[Dict]:
        """
        Attempt to detect chapter boundaries in text.
        
        Returns list of chapters with detected titles and content.
        """
        chapters = []
        current_chapter = None
        current_content = []
        
        for para in paragraphs:
            # Check if paragraph looks like a chapter heading
            is_chapter_heading = False
            for pattern in cls.CHAPTER_PATTERNS:
                if re.match(pattern, para.lower().strip()):
                    is_chapter_heading = True
                    break
            
            if is_chapter_heading:
                # Save previous chapter
                if current_chapter is not None:
                    chapters.append({
                        'title': current_chapter,
                        'content': '\n\n'.join(current_content)
                    })
                
                # Start new chapter
                current_chapter = para
                current_content = []
            else:
                # Add to current chapter
                if current_chapter is not None:
                    current_content.append(para)
                else:
                    # No chapter detected yet, treat as first chapter
                    if not chapters and not current_content:
                        current_chapter = "Chapter 1"
                    current_content.append(para)
        
        # Save final chapter
        if current_chapter is not None and current_content:
            chapters.append({
                'title': current_chapter,
                'content': '\n\n'.join(current_content)
            })
        
        # If no chapters detected, treat entire text as one chapter
        if not chapters and paragraphs:
            chapters.append({
                'title': "Chapter 1",
                'content': '\n\n'.join(paragraphs)
            })
        
        return chapters
    
    @classmethod
    def estimate_word_count(cls, text: str) -> int:
        """Estimate word count from text."""
        return len(text.split())
