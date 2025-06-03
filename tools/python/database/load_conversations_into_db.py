import json
from datetime import datetime

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

# Start a new session
def start_session(user_id=None):
    cursor.execute(
        "INSERT INTO sessions (user_id, session_start) VALUES (%s, %s) RETURNING id",
        (user_id, datetime.utcnow())
    )
    session_id = cursor.fetchone()[0]
    conn.commit()
    return session_id

# Log a conversation turn
def log_turn(session_id, user_input, agent_response, turn_order, metadata=None):
    cursor.execute(
        """
        INSERT INTO conversation_turns
        (session_id, user_input, agent_response, turn_order, input_timestamp, response_timestamp, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            session_id,
            user_input,
            agent_response,
            turn_order,
            datetime.utcnow(),  # Input timestamp
            datetime.utcnow(),  # Response timestamp (could measure actual latency)
            json.dumps(metadata or {})
        )
    )
    conn.commit()

# Example usage
session_id = start_session(user_id=None)  # Anonymous user
log_turn(
    session_id,
    "What’s the best way to store song lyrics in a database?",
    "Here’s a schema with `artists`, `songs`, and `lyric_sections` tables...",
    1,
    {"latency": 2.0, "source": "Grok 3"}
)

# Close connection
cursor.close()
conn.close()
