from sentence_transformers import SentenceTransformer

# KoSentenceBERT 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 임베딩할 문장 리스트
sentences = [
    "안녕하세요, 오늘 날씨가 좋네요.",
    "AI를 활용한 비즈니스 혁신에 관심이 있습니다.",
    "회사의 데이터 분석 역량을 강화하고 싶습니다."
]

# 문장 임베딩 생성
embeddings = model.encode(sentences)

# 결과 확인
for sentence, embedding in zip(sentences, embeddings):
    print(f"문장: {sentence}")
    print(f"임베딩 벡터 길이: {len(embedding)}")
    print(f"임베딩 벡터 값 (일부): {embedding[:5]}")
    print("-" * 50)




