-- Table for users (optional, if you want to track users)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE,          -- Optional identifier
    metadata JSONB,                        -- e.g., {"location": "US", "signup_date": "2025-01-01"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for conversation sessions
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) DEFAULT NULL,  -- Nullable if anonymous
    session_start TIMESTAMP DEFAULT NOW(),
    session_end TIMESTAMP DEFAULT NULL,
    metadata JSONB,                        -- e.g., {"device": "mobile", "intent": "query"}
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for conversation turns
CREATE TABLE conversation_turns (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    user_input TEXT NOT NULL,              -- User’s prompt or question
    agent_response TEXT NOT NULL,          -- AI’s response
    turn_order INTEGER,                    -- Order of this turn in the session
    input_timestamp TIMESTAMP DEFAULT NOW(),
    response_timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB,                        -- e.g., {"confidence": 0.95, "latency": 0.2}
    created_at TIMESTAMP DEFAULT NOW()
);

--- users: Tracks unique users (optional, depending on whether you need user-specific data).
--- sessions: Groups turns into a single conversation session (e.g., a chat thread).
--- conversation_turns: Stores each user-agent exchange with a sequence (turn_order).
