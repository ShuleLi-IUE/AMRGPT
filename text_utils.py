from nltk.tokenize import sent_tokenize

def split_text(paragraphs, chunk_size=300, overlap_size=100):
    """按指定 chunk_size 和 overlap_size 交叠割文本"""
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i= 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap+chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks

def split_text_with_pages(paragraphs, pages, chunk_size=300, overlap_size=100):
    """按指定 chunk_size 和 overlap_size 交叠割文本"""
    
    sentences = []
    pages_a = []
    j = 0
    for p in paragraphs:
        ss = [s.strip() for s in sent_tokenize(p)]
        sentences.extend(ss)
        pages_a.extend([pages[j]] * len(ss))
        j += 1

    i=0   
    chunks = []
    pages_match = []
    while i < len(sentences):
        chunk = sentences[i]
        page_now = pages_a[i]
        overlap = ''
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap+chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        pages_match.append(page_now)
        i = next
        
    return chunks, pages_match