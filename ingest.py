from pypdf import PdfReader

def ingest_pdf(path: str) -> list[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages

def chunk_text(text: str, chunk_size=800, overlap=150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def build_chunks(pages: list[str], chunk_size=800, overlap=150) -> list[str]:
    all_text = "\n\n".join(pages)
    return chunk_text(all_text, chunk_size=chunk_size, overlap=overlap)
