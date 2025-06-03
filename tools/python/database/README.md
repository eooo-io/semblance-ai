### Workflow

1. Convert lyrics to `.txt` (if not already done).
2. Load into PostgreSQL with `artists`, `songs`, and optionally `lyric_sections`.
3. Export to Elasticsearch, either as pre-chunked sections or dynamically split chunks.
4. Use Elasticsearch for fast retrieval in your RAG pipeline.

This structure keeps your data organized and scalable while optimizing for RAG use cases. Let me know if your lyrics have a specific format I should adjust for!

* Normalization: Standardize artist names (e.g., "The Beatles" vs. "Beatles") to avoid duplicates.
* Section Detection: If your lyrics don’t have explicit [Verse] tags, use heuristics (e.g., blank lines) to split them.
* Metadata Enrichment: Add fields like language or mood to songs.metadata if relevant for RAG.
* Elasticsearch Mapping: Define a mapping with text fields for section_text/chunk_text and keyword fields for title, artist, etc., for exact matches.
