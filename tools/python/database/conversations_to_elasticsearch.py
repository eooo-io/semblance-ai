from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])
conn = psycopg2.connect(...)  # Same as above
cursor = conn.cursor()

cursor.execute("""
    SELECT session_id, user_input, agent_response, turn_order, input_timestamp, metadata
    FROM conversation_turns
""")
for session_id, user_input, agent_response, turn_order, input_timestamp, metadata in cursor.fetchall():
    es.index(
        index="conversations",
        id=f"{session_id}_{turn_order}",
        body={
            "session_id": session_id,
            "user_input": user_input,
            "agent_response": agent_response,
            "turn_order": turn_order,
            "input_timestamp": str(input_timestamp),
            "metadata": metadata
        }
    )

conn.close()


# Index: Use a single index (conversations) with fields like user_input and agent_response as text for full-text search.
# Chunking: If responses are long, you could split agent_response into smaller chunks, similar to lyrics or books.
