"""Naive Dense-Only Baseline RAG Pipeline for controlled ablation comparison."""

from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from src.config import settings
from src.agent.nodes import get_llm

BASELINE_COLLECTION = "baseline_naive_dense"


class NaiveBaselineRAG:
    def __init__(self, client: QdrantClient = None):
        self.client = client or QdrantClient(":memory:")
        self.embed_model = TextEmbedding(model_name=settings.DENSE_EMBED_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self.is_indexed = False

    def build_baseline_index(self, markdown_dir: str = "data/processed_md"):
        """Split text into fixed 500-character chunks and index into dense-only Qdrant collection."""
        self.client.recreate_collection(
            collection_name=BASELINE_COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

        md_files = list(Path(markdown_dir).glob("*.md"))
        all_chunks = []
        for mf in md_files:
            with open(mf, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self.text_splitter.split_text(text)
            for idx, c in enumerate(chunks):
                all_chunks.append({
                    "id": f"{mf.stem}_{idx}",
                    "text": c,
                    "source": mf.name
                })

        print(f"[BASELINE] Split {len(md_files)} documents into {len(all_chunks)} naive 500-char chunks.")

        # Batch embed and upsert
        batch_size = 128
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [b["text"] for b in batch]
            vectors = [v.tolist() for v in self.embed_model.embed(texts)]

            points = [
                models.PointStruct(
                    id=i + idx,
                    vector=vec,
                    payload={"text": b["text"], "source": b["source"]}
                )
                for idx, (b, vec) in enumerate(zip(batch, vectors))
            ]
            self.client.upsert(collection_name=BASELINE_COLLECTION, points=points)

        self.is_indexed = True
        print(f"[BASELINE] Indexed {len(all_chunks)} points in dense-only baseline.")

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Dense-only retrieval with no grading, no rewriting, and direct synthesis."""
        if not self.is_indexed:
            self.build_baseline_index()

        q_vec = list(self.embed_model.embed([question]))[0].tolist()
        hits = self.client.query_points(
            collection_name=BASELINE_COLLECTION,
            query=q_vec,
            limit=top_k,
            with_payload=True
        ).points

        contexts = [h.payload.get("text", "") for h in hits]

        # Pass directly to LLM without grading
        llm = get_llm(settings.SYNTHESIS_LLM_MODEL, temperature=0.0, max_tokens=1024)
        if llm:
            prompt = f"Answer the following question based only on the provided context:\n\nContext:\n{' '.join(contexts)}\n\nQuestion: {question}"
            try:
                answer = llm.invoke(prompt).content.strip()
            except Exception as e:
                answer = f"Baseline synthesis fallback: {contexts[0][:200]}..."
        else:
            answer = f"Dense retrieval top context: {contexts[0][:200]}..." if contexts else "No context found."

        return {
            "question": question,
            "answer": answer,
            "retrieved_contexts": contexts,
        }
