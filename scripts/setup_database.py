#!/usr/bin/env python3
"""
Database setup script for ATOM-GPT
Run this to initialize the database with migrations
"""
import os
import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from database.migrator import DatabaseMigrator
from database.models import DatabaseManager

def setup_database():
    """Initialize database with migrations"""
    print("🚀 Setting up ATOM-GPT Database...")
    
    # Ensure database directory exists
    db_dir = Path("backend/database")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = "backend/database/atom_gpt.db"
    
    # Run migrations
    print("📋 Running database migrations...")
    migrator = DatabaseMigrator(db_path)
    count, executed = migrator.migrate()
    
    if count > 0:
        print(f"✅ Database setup complete! Executed {count} migrations:")
        for migration in executed:
            print(f"   📄 {migration}")
    else:
        print("📋 Database already up to date")
    
    # Verify database
    print("\n🔍 Verifying database...")
    db = DatabaseManager(db_path)
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['users', 'conversations', 'messages', 'user_settings', 'api_usage', 'migrations']
        missing_tables = [table for table in expected_tables if table not in tables]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
        else:
            print("✅ All tables created successfully")
    
    print(f"\n💾 Database location: {os.path.abspath(db_path)}")
    print("🎉 Ready to use ATOM-GPT with persistent storage!")
    
    return db_path

if __name__ == "__main__":
    setup_database()