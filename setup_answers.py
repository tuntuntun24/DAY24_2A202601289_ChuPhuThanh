"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import COLLECTION_NAME, RERANK_TOP_K

    # Reuse the persisted Qdrant collection after an interrupted setup run.
    search = HybridSearch()
    try:
        if search.dense.client.collection_exists(COLLECTION_NAME):
            points, _ = search.dense.client.scroll(
                collection_name=COLLECTION_NAME, limit=1000, with_payload=True,
            )
            indexed_chunks = [
                {"text": point.payload["text"], "metadata": dict(point.payload or {})}
                for point in points if point.payload and point.payload.get("text")
            ]
            if indexed_chunks:
                search.bm25.index(indexed_chunks)
                reranker = CrossEncoderReranker(model_name="lexical")
                print(f"✓ Reusing {len(indexed_chunks)} indexed chunks from Qdrant")
                return search, reranker, RERANK_TOP_K
    except Exception as exc:
        print(f"  ⚠️  Could not reuse Qdrant index ({exc}); rebuilding it.")

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })

    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks (M5 not implemented or no API key)")

    print("\n[2/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    print("\n[3/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker(model_name="lexical")
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int) -> tuple[str, list[str]]:
    from config import OPENAI_API_KEY

    results = search.search(q)
    docs    = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            ctx = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"},
                ],
            )
            return resp.choices[0].message.content, contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts


def run_queries(items: list[dict], search, reranker, top_k: int) -> list[tuple[str, list[str]]]:
    """Batch retrieval once, then generate answers sequentially."""
    from config import OPENAI_API_KEY
    from openai import OpenAI

    questions = [item["question"] for item in items]
    retrieved_batches = search.search_batch(questions)
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    outputs = []
    for index, (question, retrieved) in enumerate(zip(questions, retrieved_batches), 1):
        documents = [
            {"text": result.text, "score": result.score, "metadata": result.metadata}
            for result in retrieved
        ]
        reranked = reranker.rerank(question, documents, top_k=top_k)
        contexts = [result.text for result in reranked] if reranked else [r.text for r in retrieved[:3]]
        if client and contexts:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có, nói 'Không tìm thấy thông tin.'."},
                        {"role": "user", "content": f"Context:\n{'\n\n'.join(contexts)}\n\nCâu hỏi: {question}"},
                    ],
                )
                answer = response.choices[0].message.content or "Không tìm thấy thông tin."
            except Exception as exc:
                print(f"  ⚠️  LLM generation failed for question {index}: {exc}")
                answer = contexts[0] if contexts else "Không tìm thấy thông tin."
        else:
            answer = contexts[0] if contexts else "Không tìm thấy thông tin."
        outputs.append((answer, contexts))
        if index % 10 == 0:
            print(f"  [{index}/{len(items)}] generated")
    return outputs


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    if not check_day18_files():
        sys.exit(1)

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    try:
        search, reranker, top_k = build_pipeline()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("→ Đảm bảo bạn đã copy src/ từ Day 18 và đã pip install -r requirements.txt")
        sys.exit(1)

    print(f"\nRunning {len(test_set)} queries...")
    answers = []
    t_start = time.time()

    generated = run_queries(test_set, search, reranker, top_k)
    for i, (item, (answer, contexts)) in enumerate(zip(test_set, generated)):
        answers.append({
            "id":           item["id"],
            "distribution": item["distribution"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
