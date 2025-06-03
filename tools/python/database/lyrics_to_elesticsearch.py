from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])
conn = psycopg2.connect(...)  # Same as above
cursor = conn.cursor()

cursor.execute("""
    SELECT s.id, s.title, a.name AS artist, ls.section_type, ls.section_text, ls.section_order
    FROM songs s
    JOIN artists a ON s.artist_id = a.id
    LEFT JOIN lyric_sections ls ON s.id = ls.song_id
""")
for song_id, title, artist, section_type, section_text, section_order in cursor.fetchall():
    es.index(
        index="lyrics",
        id=f"{song_id}_{section_order or 0}",
        body={
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "section_type": section_type or "full",
            "section_text": section_text or lyrics,  # Fallback to full lyrics if no sections
            "section_order": section_order or 0
        }
    )

conn.close()

# OR store export full lyric tests

# def split_lyrics(lyrics, max_length=500):
#     return [lyrics[i:i + max_length] for i in range(0, len(lyrics), max_length)]

# cursor.execute("SELECT id, title, artist_id, lyrics FROM songs JOIN artists ON songs.artist_id = artists.id")
# for song_id, title, artist_id, lyrics in cursor.fetchall():
#     chunks = split_lyrics(lyrics)
#     for i, chunk in enumerate(chunks):
#         es.index(
#             index="lyrics",
#             id=f"{song_id}_{i}",
#             body={
#                 "song_id": song_id,
#                 "title": title,
#                 "artist_id": artist_id,
#                 "chunk_text": chunk,
#                 "chunk_order": i
#             }
#         )
