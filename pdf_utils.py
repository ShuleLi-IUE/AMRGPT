from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pdfplumber

def extract_text_from_pdf_pdfminer(filename, page_numbers=None, min_line_length=10):
    """从 PDF 文件中（按指定页码）提取文字"""
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

def extract_text_from_pdf_pdfplumber_with_pages(filename, page_numbers=None, min_line_length=10):
    pdf = pdfplumber.open(filename)
    paragraphs = []
    pages_record = []
    buffer = ''
    for page in pdf.pages:
        if page_numbers is not None and page.page_number not in page_numbers:
            continue
        lines = page.extract_text().split("\n")
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (' '+text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                paragraphs.append(buffer)
                pages_record.append(page.page_number)
                buffer = ''
    if buffer:
        paragraphs.append(buffer)
        pages_record.append(page.page_number)
    pdf.close()
    return paragraphs, pages_record


# def extract_text_from_pdf(filename,page_numbers=None,min_line_length=10,engine="pdfminer"):
#     if engine == "pdfminer":
#         return extract_text_from_pdf_pdfminer(filename, page_numbers, min_line_length)
#     else:
#         return extract_text_from_pdf_pdfplumber_with_pages(filename, page_numbers, min_line_length)
