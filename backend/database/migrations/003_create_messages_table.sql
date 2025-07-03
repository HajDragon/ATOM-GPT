-- Messages table for storing chat/completion content
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,  -- Format: msg_20241201_123456_abc
    conversation_id TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    enhanced BOOLEAN DEFAULT FALSE,
    tokens_used INTEGER,
    processing_time REAL,  -- In seconds
    model_used VARCHAR(100),
    temperature REAL,
    top_p REAL,
    repetition_penalty REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_enhanced ON messages(enhanced);