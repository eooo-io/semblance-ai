import psycopg2
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Connect to PostgreSQL
conn = psycopg2.connect(...)  # Same as above
cursor = conn.cursor()

# Fetch books or chunks
cursor.execute("SELECT id, title, content FROM books")  # Or use book_chunks table
for book_id, title, content in cursor.fetchall():
    # Split content into chunks if not already done
    chunks = split_into_chunks(content, max_length=1000)
    for i, chunk in enumerate(chunks):
        es.index(
            index="books",
            id=f"{book_id}_{i}",
            body={
                "book_id": book_id,
                "title": title,
                "chunk_text": chunk,
                "chunk_order": i
            }
        )

conn.close()
