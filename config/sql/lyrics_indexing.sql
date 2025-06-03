-- Index for full-text search on lyrics (optional for PostgreSQL querying)
CREATE INDEX idx_lyrics ON songs USING GIN (to_tsvector('english', lyrics));

-- Index for faster joins
CREATE INDEX idx_song_artist ON songs (artist_id);
CREATE INDEX idx_song_album ON songs (album_id);
CREATE INDEX idx_lyric_section_song ON lyric_sections (song_id);
