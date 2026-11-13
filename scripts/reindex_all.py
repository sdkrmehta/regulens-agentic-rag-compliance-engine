import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.html_to_markdown import convert_rbi_html_to_markdown
from src.ingestion.chunker import TwoTierChunker, DOC_METADATA_MAP
from src.retrieval.vector_store import RBIHybridVectorStore

files = [
    ('kyc_html.html', 'KYC', 'data/processed_md/rbi_kyc_master_direction.md'),
    ('dl_html.html', 'DIGITAL_LENDING', 'data/processed_md/rbi_digital_lending_guidelines.md'),
    ('cyb_html.html', 'CYBER_RESILIENCE', 'data/processed_md/rbi_cyber_resilience_master_direction.md')
]

for html_file, cat, out_path in files:
    md = convert_rbi_html_to_markdown(Path(html_file), cat)
    out_p = Path(out_path)
    with open(out_p, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Exported {out_path} ({len(md)} chars)")

chunker = TwoTierChunker(
    parent_token_limit=1500,
    child_token_size=256,
    child_token_overlap=32,
)

all_parents = {}
all_children = []

for html_file, cat, md_path in files:
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    parents, children = chunker.process_document(
        markdown_text=text,
        doc_category=cat,
        doc_metadata=DOC_METADATA_MAP[cat]
    )
    all_parents.update(parents)
    all_children.extend(children)
    print(f"[{cat}] {len(parents)} parents, {len(children)} children")

print(f"Total: {len(all_parents)} parents, {len(all_children)} children.")

# Safe, atomic re-index into Qdrant.
# The parent docstore is written to disk ONLY inside on_success, which fires
# after the staging collection is verified and promoted to production.
# This guarantees the disk file and Qdrant are always in sync — if Qdrant
# indexing fails, the old parent_docstore.json is left untouched.
store = RBIHybridVectorStore(docstore_path="data/parent_docstore.json")

def _persist_docstore():
    chunker.save_parent_docstore(all_parents, "data/parent_docstore.json")
    # Invalidate the in-process singleton so the next read picks up the new file.
    import src.retrieval.vector_store as _vs
    _vs._PARENT_DOCSTORE = None

store.atomic_reindex(all_children, batch_size=64, on_success=_persist_docstore)
print("Safe atomic re-indexing complete!")
