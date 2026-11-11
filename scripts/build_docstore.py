"""Script to process Markdown documents through TwoTierChunker and populate parent_docstore.json."""

from pathlib import Path
from src.ingestion.chunker import TwoTierChunker, DOC_METADATA_MAP

CORPUS_FILES = [
    {
        "filepath": "data/processed_md/rbi_kyc_master_direction.md",
        "category": "KYC",
        "metadata": DOC_METADATA_MAP["KYC"]
    },
    {
        "filepath": "data/processed_md/rbi_digital_lending_guidelines.md",
        "category": "DIGITAL_LENDING",
        "metadata": DOC_METADATA_MAP["DIGITAL_LENDING"]
    },
    {
        "filepath": "data/processed_md/rbi_cyber_resilience_master_direction.md",
        "category": "CYBER_RESILIENCE",
        "metadata": DOC_METADATA_MAP["CYBER_RESILIENCE"]
    }
]


def main():
    chunker = TwoTierChunker(
        parent_token_limit=1500,
        child_token_size=256,
        child_token_overlap=32,
    )

    all_parents = {}
    all_children = []

    for item in CORPUS_FILES:
        path = Path(item["filepath"])
        if not path.exists():
            print(f"[ERROR] Processed markdown file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        parents, children = chunker.process_document(
            markdown_text=text,
            doc_category=item["category"],
            doc_metadata=item["metadata"]
        )

        print(f"[{item['category']}] Generated {len(parents)} parents and {len(children)} child chunks.")
        all_parents.update(parents)
        all_children.extend(children)

    # Save to parent_docstore.json
    chunker.save_parent_docstore(all_parents, "data/parent_docstore.json")
    print(f"\n[TOTAL] Successfully created {len(all_parents)} parent sections and {len(all_children)} child chunks!")


if __name__ == "__main__":
    main()
