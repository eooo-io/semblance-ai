-- ----------------------------------------------------------------------------
-- Books
-- ----------------------------------------------------------------------------

CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,           -- Book title
    author VARCHAR(255),                   -- Author (optional)
    file_name VARCHAR(255) UNIQUE,         -- Original .epub file name or .txt file name
    content TEXT NOT NULL,                 -- Full text content of the book
    metadata JSONB,                        -- Additional metadata (e.g., chapters, publication date)
    created_at TIMESTAMP DEFAULT NOW()     -- Timestamp for record creation
);

-- ----------------------------------------------------------------------------
-- Song Lyrics
-- ----------------------------------------------------------------------------

-- Table for artists
CREATE TABLE artists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    metadata JSONB,                        -- e.g., nationality, formation year
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for albums (optional, if you have album data)
CREATE TABLE albums (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist_id INTEGER REFERENCES artists(id),
    release_year INTEGER,
    metadata JSONB,                        -- e.g., genre, label
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for songs
CREATE TABLE songs (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist_id INTEGER REFERENCES artists(id),
    album_id INTEGER REFERENCES albums(id) DEFAULT NULL,  -- Nullable if no album
    lyrics TEXT NOT NULL,                  -- Full lyrics as a single text blob
    file_name VARCHAR(255) UNIQUE,         -- Source file name (if applicable)
    metadata JSONB,                        -- e.g., genre, duration, writer
    created_at TIMESTAMP DEFAULT NOW()
);

-- Optional: Table for lyric sections (verses, choruses, etc.)
CREATE TABLE lyric_sections (
    id SERIAL PRIMARY KEY,
    song_id INTEGER REFERENCES songs(id),
    section_type VARCHAR(50),              -- e.g., "verse", "chorus", "bridge"
    section_text TEXT NOT NULL,            -- Text of this section
    section_order INTEGER,                 -- Order within the song
    created_at TIMESTAMP DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- Transcripts (Video, Podcast etc.)
-- ----------------------------------------------------------------------------

CREATE TABLE media (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,           -- Title of the video or podcast
    media_type VARCHAR(50) NOT NULL,       -- e.g., "video", "podcast"
    source_url VARCHAR(255),               -- URL or file path (optional)
    duration INTEGER,                      -- Duration in seconds (optional)
    metadata JSONB,                        -- e.g., {"creator": "John Doe", "publish_date": "2025-01-01"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for transcripts
CREATE TABLE transcripts (
    id SERIAL PRIMARY KEY,
    media_id INTEGER REFERENCES media(id),
    full_text TEXT NOT NULL,               -- Full transcript as a single text blob
    language VARCHAR(10) DEFAULT 'en',     -- Language code, e.g., "en" for English
    transcription_method VARCHAR(50),      -- e.g., "manual", "auto-generated", "Whisper"
    metadata JSONB,                        -- e.g., {"accuracy": 0.95, "transcribed_by": "Whisper v3"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- Conversations
-- ----------------------------------------------------------------------------

-- Table for papers
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,           -- Paper title
    file_name VARCHAR(255) UNIQUE,         -- Original .pdf file name (for deduplication)
    full_text TEXT NOT NULL,               -- Full extracted text from the PDF
    abstract TEXT,                         -- Abstract text (if separable)
    publication_year INTEGER,              -- Year of publication (optional)
    source VARCHAR(255),                   -- e.g., "arXiv", "IEEE", file path, or URL
    language VARCHAR(10) DEFAULT 'en',     -- Language code, e.g., "en" for English
    metadata JSONB,                        -- e.g., {"doi": "10.1000/xyz", "keywords": ["AI", "RAG"]}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for authors (to handle multiple authors per paper)
CREATE TABLE authors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,            -- Author’s full name
    affiliation VARCHAR(255),              -- Author’s institution (optional)
    metadata JSONB,                        -- e.g., {"email": "author@university.edu"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Junction table for paper-author relationships (many-to-many)
CREATE TABLE paper_authors (
    paper_id INTEGER REFERENCES papers(id),
    author_id INTEGER REFERENCES authors(id),
    author_order INTEGER,                  -- Order of authorship (1 = first author, etc.)
    PRIMARY KEY (paper_id, author_id)
);

-- Table for paper sections (optional, for structured text like abstract, introduction, etc.)
CREATE TABLE paper_sections (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id),
    section_title VARCHAR(255),            -- e.g., "Abstract", "Introduction", "Conclusion"
    section_text TEXT NOT NULL,            -- Text content of this section
    section_order INTEGER,                 -- Order of this section in the paper
    metadata JSONB,                        -- e.g., {"page_number": 3, "word_count": 250}
    created_at TIMESTAMP DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- Conversations
-- ----------------------------------------------------------------------------

-- Table for conversation sessions
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP DEFAULT NULL,
    metadata JSONB,                        -- e.g., {"experiment_name": "Semblance_v1", "purpose": "RAG_test"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for conversation turns
CREATE TABLE conversation_turns (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    user_input TEXT NOT NULL,              -- Your prompt or question
    agent_response TEXT NOT NULL,          -- AI’s response
    ai_provider VARCHAR(50) NOT NULL,      -- e.g., "ChatGPT", "Grok", "Claude"
    model_version VARCHAR(50),             -- e.g., "gpt-4", "Grok 3", "claude-3-sonnet"
    turn_order INTEGER,                    -- Order of this turn in the session
    input_timestamp TIMESTAMP DEFAULT NOW(),
    response_timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB,                        -- e.g., {"latency": 0.2, "tokens_used": 150}
    created_at TIMESTAMP DEFAULT NOW()
);
