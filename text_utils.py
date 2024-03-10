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
