-- Indexes for performance
CREATE INDEX idx_session_id ON conversation_turns (session_id);
CREATE INDEX idx_input_timestamp ON conversation_turns (input_timestamp);
CREATE INDEX idx_user_input ON conversation_turns USING GIN (to_tsvector('english', user_input));
CREATE INDEX idx_agent_response ON conversation_turns USING GIN (to_tsvector('english', agent_response));

--- Use GIN indexes for full-text search on user_input and agent_response if you’ll query them directly in PostgreSQL
