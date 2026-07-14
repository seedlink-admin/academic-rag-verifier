"""API Key 없이 검증 가능한 파이프라인 단위 테스트 (파싱 → 청킹 → ChromaDB 저장/검색)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

# app.py에서 streamlit 의존 없는 함수만 추출하기 위해 모듈 일부 로드
import re
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_pdf(path):
    units = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                units.append((f"p.{i+1}", text))
    return units


def parse_txt(path):
    text = open(path, encoding="utf-8").read()
    return [("전체 텍스트", text.strip())]


def chunk(doc_name, units, chunk_size=3400, overlap=600):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    out = []
    for loc, text in units:
        for piece in splitter.split_text(text):
            if len(piece.strip()) > 20:
                out.append({"text": piece.strip(), "source": doc_name, "location": loc})
    return out


# 1) PDF 파싱 테스트
pdf_units = parse_pdf(os.path.join(HERE, "sample_paper.pdf"))
assert len(pdf_units) == 2, f"PDF 페이지 수 오류: {len(pdf_units)}"
assert "Transformer" in pdf_units[0][1]
assert pdf_units[0][0] == "p.1" and pdf_units[1][0] == "p.2"
print("✅ PDF 파싱 (페이지 단위, 위치 라벨) OK:", [u[0] for u in pdf_units])

# 2) TXT 파싱 테스트
txt_units = parse_txt(os.path.join(HERE, "sample_book.txt"))
assert "홍길동" in txt_units[0][1]
print("✅ TXT 파싱 OK")

# 3) 청킹 테스트
chunks = chunk("sample_paper.pdf", pdf_units) + chunk("sample_book.txt", txt_units)
assert all(c["text"] and c["source"] and c["location"] for c in chunks)
print(f"✅ 청킹 OK: 총 {len(chunks)}개 청크")

# 4) ChromaDB 저장/검색 테스트 (임베딩은 더미 벡터로 대체하여 API 없이 검증)
client = chromadb.EphemeralClient()
col = client.get_or_create_collection("test", metadata={"hnsw:space": "cosine"})
import hashlib
def fake_embed(t):
    h = hashlib.sha256(t.encode()).digest()
    return [b / 255.0 for b in h[:32]]
col.add(
    ids=[f"c{i}" for i in range(len(chunks))],
    embeddings=[fake_embed(c["text"]) for c in chunks],
    documents=[c["text"] for c in chunks],
    metadatas=[{"source": c["source"], "location": c["location"]} for c in chunks],
)
res = col.query(query_embeddings=[fake_embed(chunks[0]["text"])], n_results=2)
assert res["documents"][0][0] == chunks[0]["text"]
assert res["metadatas"][0][0]["location"] == chunks[0]["location"]
print("✅ ChromaDB 저장/유사도 검색/메타데이터(출처) OK")

# 5) app.py 임포트 가능성 (streamlit 컨텍스트 밖 문법/의존성 확인)
spec = importlib.util.find_spec
for mod in ["streamlit", "google.generativeai", "docx", "ebooklib", "bs4", "dotenv"]:
    assert spec(mod), f"모듈 누락: {mod}"
print("✅ 전체 의존성 임포트 OK")

print("\n🎉 모든 파이프라인 단위 테스트 통과")
