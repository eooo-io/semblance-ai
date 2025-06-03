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

--- artists: Stores unique artists with optional metadata (e.g., {"nationality": "American", "genre": "Rock"}).
--- albums: Links songs to albums (optional if your data doesn’t include albums).
--- songs: Holds the full lyrics and core metadata for each song.
--- lyric_sections: Breaks lyrics into smaller, labeled chunks (e.g., verse 1, chorus) for finer-grained retrieval in RAG.
