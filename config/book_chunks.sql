CREATE TABLE book_chunks (
    id SERIAL PRIMARY KEY,
    book_id INTEGER REFERENCES books(id),
    chunk_text TEXT NOT NULL,
    chunk_order INTEGER,                   -- Order of the chunk in the book
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_book_id ON book_chunks (book_id);
