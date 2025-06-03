import json
import os
import re

import psycopg2

# Database connection
conn = psycopg2.connect(
    dbname="your_db",
    user="your_user",
    password="your_password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Directory with .txt files
txt_dir = "/path/to/your/lyrics/files"

# Function to parse lyrics into sections (basic example)
def parse_lyrics(text):
    sections = []
    current_section = {"type": None, "text": ""}
    lines = text.splitlines()

    for line in lines:
        section_match = re.match(r'\[(.*?)\]', line.strip())
        if section_match:  # New section detected, e.g., [Verse 1]
            if current_section["text"]:
                sections.append(current_section)
            current_section = {"type": section_match.group(1), "text": ""}
        elif line.strip():  # Non-empty line
            current_section["text"] += line + "\n"
    if current_section["text"]:  # Add the last section
        sections.append(current_section)
    return sections

# Process each .txt file
for txt_file in os.listdir(txt_dir):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(txt_dir, txt_file)

        # Read full lyrics
        with open(file_path, "r", encoding="utf-8") as f:
            lyrics = f.read()

        # Extract title and artist (e.g., from file name: "Artist - Title.txt")
        file_base = txt_file.replace(".txt", "")
        artist_name, song_title = file_base.split(" - ", 1) if " - " in file_base else ("Unknown", file_base)

        # Insert artist
        cursor.execute(
            "INSERT INTO artists (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
            (artist_name,)
        )
        artist_id = cursor.fetchone()[0] if cursor.rowcount > 0 else cursor.execute(
            "SELECT id FROM artists WHERE name = %s", (artist_name,)
        ) or cursor.fetchone()[0]

        # Insert song
        metadata = {"source": "txt_file", "file_size": os.path.getsize(file_path)}
        cursor.execute(
            """
            INSERT INTO songs (title, artist_id, lyrics, file_name, metadata)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (song_title, artist_id, lyrics, txt_file, json.dumps(metadata))
        )
        song_id = cursor.fetchone()[0]

        # Parse and insert lyric sections (optional)
        sections = parse_lyrics(lyrics)
        for i, section in enumerate(sections):
            cursor.execute(
                """
                INSERT INTO lyric_sections (song_id, section_type, section_text, section_order)
                VALUES (%s, %s, %s, %s)
                """,
                (song_id, section["type"], section["text"].strip(), i)
            )

# Commit and close
conn.commit()
cursor.close()
conn.close()

# Parsing: The parse_lyrics function assumes lyrics might have section headers like [Verse 1]. Adjust it based on your .txt format.
# File Naming: Assumes files are named like Artist - Title.txt. Modify the split logic if your naming differs.
# Metadata: Add more fields (e.g., genre, year) to metadata if available.
