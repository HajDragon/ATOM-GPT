CREATE TBALES IF NOT EXISTS Conversations(
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title VARCHAR(200) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_message_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE,
    conversation_type VARCHAR(20) DEFAULT 'chat' CHECK (conversation_type IN ('chat', 'completion')),
    FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
);

CREATE INDEX idx_conversations_user_id ON Conversations(user_id);
CREATE INDEX idx_conversations_title ON Conversations(updated_at);
CREATE INDEX idx_conversations_last_message_at ON Conversations(last_message_at);
CREATE INDEX idx_conversations_message_count ON Conversations(conversation_type);