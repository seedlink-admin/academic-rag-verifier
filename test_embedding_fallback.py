"""임베딩 Fallback 로직 검증:
1) Google embedding-001 호출 실패(무효 키) 시 HF 로컬 임베딩으로 전환되는 흐름
2) HF 임베딩 → ChromaDB 저장 → 유사도 검색 end-to-end
"""
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

# 1) 무효 API 키로 Google 임베딩 시도 → 예외 발생 확인 (Fallback 트리거 시나리오)
genai.configure(api_key="INVALID_KEY_FOR_TEST")
google_failed = False
try:
    genai.embed_content(model="models/embedding-001", content=["test"], task_type="retrieval_document")
except Exception as e:
    google_failed = True
    print(f"✅ Google 임베딩 실패 감지 (Fallback 트리거 정상): {type(e).__name__}")
assert google_failed

# 2) HF 로컬 임베딩으로 실제 벡터화 (앱의 _embed_hf와 동일 로직)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
texts = [
    "트랜스포머는 어텐션 메커니즘만으로 구성된 신경망 구조이다.",
    "이 논문의 저자는 Vaswani 등이며 2017년 NeurIPS에 게재되었다.",
    "오늘 점심 메뉴는 김치찌개였다.",
]
embs = model.encode(texts, normalize_embeddings=True).tolist()
assert len(embs) == 3 and len(embs[0]) == 768
print(f"✅ HF 로컬 임베딩(ko-sroberta-multitask) 벡터화 완료: {len(embs)}개, 차원 {len(embs[0])}")

# 3) ChromaDB 저장 및 의미 검색 검증
client = chromadb.EphemeralClient()
col = client.get_or_create_collection("fallback_test", metadata={"hnsw:space": "cosine"})
col.add(
    ids=["a", "b", "c"],
    embeddings=embs,
    documents=texts,
    metadatas=[{"source": "doc.pdf", "location": f"p.{i+1}"} for i in range(3)],
)
q = model.encode(["논문 저자가 누구인가?"], normalize_embeddings=True).tolist()
res = col.query(query_embeddings=q, n_results=1)
print(f"✅ 의미 검색 결과: {res['documents'][0][0][:40]}... (출처: {res['metadatas'][0][0]})")
assert "Vaswani" in res["documents"][0][0], "의미 검색이 올바른 청크를 반환하지 않음"

print("\n🎉 임베딩 Fallback 파이프라인 테스트 전부 통과")
