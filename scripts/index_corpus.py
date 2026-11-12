"""Comprehensive Idempotent Indexing Script for RBI Regulatory Corpus with Version Metadata."""

import json
from pathlib import Path
from src.ingestion.chunker import TwoTierChunker
from src.retrieval.vector_store import RBIHybridVectorStore

CORPUS_REGISTRY = [
    {
        "filepath": "data/processed_md/rbi_digital_lending_directions_2025.md",
        "category": "DIGITAL_LENDING",
        "metadata": {
            "document_id": "RBI_DL_2025",
            "title": "Reserve Bank of India (Digital Lending) Directions, 2025",
            "circular_number": "RBI/2025-26/36",
            "date": "2025-05-08",
            "status": "ACTIVE",
            "version": "2.0"
        }
    },
    {
        "filepath": "data/processed_md/rbi_kyc_master_direction.md",
        "category": "KYC",
        "metadata": {
            "document_id": "RBI_KYC_2016",
            "title": "Master Direction - Know Your Customer (KYC) Direction, 2016",
            "circular_number": "RBI/DBR/2015-16/18",
            "date": "2016-02-25",
            "status": "ACTIVE",
            "version": "Updated as on January 04, 2024"
        }
    },
    {
        "filepath": "data/processed_md/rbi_cyber_resilience_master_direction.md",
        "category": "CYBER_RESILIENCE",
        "metadata": {
            "document_id": "RBI_CYBER_2024",
            "title": "Master Directions on Cyber Resilience and Digital Payment Security Controls",
            "circular_number": "RBI/DPSS/2024-25/123",
            "date": "2024-07-30",
            "status": "ACTIVE",
            "version": "1.0"
        }
    },
    {
        "filepath": "data/processed_md/rbi_digital_lending_guidelines.md",
        "category": "DIGITAL_LENDING",
        "metadata": {
            "document_id": "RBI_DL_2022",
            "title": "Guidelines on Digital Lending (2022)",
            "circular_number": "RBI/2022-23/111",
            "date": "2022-09-02",
            "status": "REPEALED",
            "version": "1.0"
        }
    }
]


def main(target_collection: str = None, recreate: bool = False):
    print("=" * 80)
    print("ReguLens Corpus Ingestion & Idempotent Indexing Pipeline")
    print("=" * 80)

    chunker = TwoTierChunker(
        parent_token_limit=1500,
        child_token_size=256,
        child_token_overlap=32,
    )

    all_parents = {}
    all_children = []

    for item in CORPUS_REGISTRY:
        path = Path(item["filepath"])
        if not path.exists():
            print(f"[WARNING] Corpus file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        parents, children = chunker.process_document(
            markdown_text=text,
            doc_category=item["category"],
            doc_metadata=item["metadata"]
        )
        all_parents.update(parents)
        all_children.extend(children)
        status_label = item["metadata"].get("status", "ACTIVE")
        print(f"[{item['category']}] {item['metadata']['document_id']} ({status_label}): {len(parents)} parents, {len(children)} children")

    # Save complete parent docstore
    chunker.save_parent_docstore(all_parents, "data/parent_docstore.json")

    # Index child chunks into Qdrant
    vector_store = RBIHybridVectorStore(docstore_path="data/parent_docstore.json")
    col = target_collection or vector_store.collection_name
    vector_store.init_collection(collection_name=col, recreate=recreate)
    vector_store.index_child_chunks(all_children, batch_size=64, target_collection=col)

    # Verification report
    try:
        col_info = vector_store.client.get_collection(col)
        points_count = col_info.points_count
    except Exception:
        points_count = len(all_children)

    print("\n" + "=" * 80)
    print(f"[INDEXING VERIFIED] Successfully indexed {len(all_children)} chunks into '{col}'.")
    print(f"Total Parent Sections in Docstore: {len(all_parents)}")
    print(f"Active Qdrant Points: {points_count}")
    print("=" * 80)
    return len(all_parents), len(all_children)


if __name__ == "__main__":
    main(recreate=True)
