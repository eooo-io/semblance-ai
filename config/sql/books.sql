CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,           -- Book title
    author VARCHAR(255),                   -- Author (optional)
    file_name VARCHAR(255) UNIQUE,         -- Original .epub file name or .txt file name
    content TEXT NOT NULL,                 -- Full text content of the book
    metadata JSONB,                        -- Additional metadata (e.g., chapters, publication date)
    created_at TIMESTAMP DEFAULT NOW()     -- Timestamp for record creation
);

-- Add an index on the content for faster searches (optional, depending on your queries)
CREATE INDEX idx_content ON books USING GIN (to_tsvector('english', content));

--- TEXT for content: Use the TEXT type to store the full extracted text from each .txt file. It has no practical size limit in PostgreSQL.
--- JSONB for metadata: Store structured metadata (e.g., chapter titles, page counts) in a JSONB column for flexibility and efficient querying.
--- Indexing: The GIN index with to_tsvector is optional but useful if you plan to do full-text search directly in PostgreSQL before moving to Elasticsearch.
