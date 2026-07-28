from __future__ import annotations

from hashlib import blake2b

from .models import DocumentChunk
from .text_processing import split_text


def build_parent_child_chunks(
    chunks: list[DocumentChunk],
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
    parents: list[DocumentChunk] = []
    children: list[DocumentChunk] = []
    for parent_index, chunk in enumerate(chunks):
        parent = normalize_parent(chunk, parent_index)
        parents.append(parent)
        child_texts = split_text(parent.text, child_chunk_size, child_chunk_overlap)
        if not child_texts:
            child_texts = [parent.text]
        added_for_parent = 0
        for child_index, child_text in enumerate(child_texts):
            if len(child_text.strip()) < 20:
                continue
            children.append(make_child_chunk(parent, child_text, child_index))
            added_for_parent += 1
        if added_for_parent == 0 and parent.text.strip():
            children.append(make_child_chunk(parent, parent.text, 0))
    return parents, children


def normalize_parent(chunk: DocumentChunk, index: int) -> DocumentChunk:
    parent_id = chunk.parent_id or make_parent_id(chunk, index)
    metadata = dict(chunk.metadata)
    metadata.setdefault("parent_id", parent_id)
    metadata.setdefault("chunk_role", "parent")
    return DocumentChunk(
        chunk_id=parent_id,
        source=chunk.source,
        page=chunk.page,
        text=chunk.text,
        modality=chunk.modality,
        parent_id=parent_id,
        chunk_role="parent",
        metadata=metadata,
    )


def make_child_chunk(parent: DocumentChunk, text: str, child_index: int) -> DocumentChunk:
    child_id = make_child_id(parent.parent_id, child_index, text)
    metadata = dict(parent.metadata)
    metadata.update(
        {
            "parent_id": parent.parent_id,
            "parent_source": parent.source,
            "parent_page": parent.page,
            "child_index": child_index,
            "chunk_role": "child",
        }
    )
    return DocumentChunk(
        chunk_id=child_id,
        source=parent.source,
        page=parent.page,
        text=text,
        modality=parent.modality,
        parent_id=parent.parent_id,
        chunk_role="child",
        metadata=metadata,
    )


def make_parent_id(chunk: DocumentChunk, index: int) -> str:
    digest = blake2b(
        f"parent:{chunk.source}:{chunk.page}:{index}:{chunk.modality}:{chunk.text[:160]}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"{chunk.source}:{chunk.page}:parent:{index}:{digest}"


def make_child_id(parent_id: str, child_index: int, text: str) -> str:
    digest = blake2b(f"child:{parent_id}:{child_index}:{text[:120]}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{parent_id}:child:{child_index}:{digest}"
