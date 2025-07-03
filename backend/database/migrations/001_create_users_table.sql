CREATE TABLES IF NOT EXISTS USERS(
    IF INTEGER PRIMARY KEY AUTOINCREMENT,
    username(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON USERS(email);
CREATE INDEX idx_users_username ON USERS(username);

--admibin user crendentials
INSERT INTO USERS (username, email, password_hash, first_name, last_name, is_admin)
VALUES (1, 'arshia', 'arshia.admin@gmail.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewbYJw/Wy2HT0gZm', TRUE);