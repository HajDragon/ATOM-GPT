-- User settings for persistent configuration
CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    setting_key VARCHAR(100) NOT NULL,
    setting_value TEXT,
    setting_type VARCHAR(20) DEFAULT 'string' CHECK (setting_type IN ('string', 'number', 'boolean', 'json')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, setting_key)
);

CREATE INDEX idx_user_settings_user_id ON user_settings(user_id);
CREATE INDEX idx_user_settings_key ON user_settings(setting_key);

-- Insert default settings for admin user
INSERT OR IGNORE INTO user_settings (user_id, setting_key, setting_value, setting_type) VALUES
(1, 'tokens', '60', 'number'),
(1, 'temperature', '0.7', 'number'),
(1, 'top_p', '0.9', 'number'),
(1, 'repetition_penalty', '1.1', 'number'),
(1, 'theme', 'dark', 'string'),
(1, 'auto_save', 'true', 'boolean'),
(1, 'enhance_enabled', 'true', 'boolean'),
(1, 'lm_studio_url', 'http://localhost:1234', 'string');