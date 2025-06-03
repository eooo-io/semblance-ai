import json
import os

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

# Directory containing .txt files
txt_dir = "/path/to/your/txt/files"

# Iterate over .txt files
for txt_file in os.listdir(txt_dir):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(txt_dir, txt_file)

        # Read the text content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Example metadata (customize as needed)
        metadata = {
            "original_format": "epub",
            "converted_to": "txt",
            "file_size": os.path.getsize(file_path)
        }

        # Insert into PostgreSQL
        cursor.execute(
            """
            INSERT INTO books (title, file_name, content, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_name) DO NOTHING;
            """,
            (txt_file.replace(".txt", ""), txt_file, content, json.dumps(metadata))
        )

# Commit and close
conn.commit()
cursor.close()
conn.close()

# Error Handling: Add try-except blocks if your .txt files might have encoding issues or other problems.
# Metadata: Extract metadata from the .epub before conversion (e.g., using ebooklib) if you want richer data like chapter titles or authors.
