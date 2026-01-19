from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    将长文本分块以便向量化与检索。
    """
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

