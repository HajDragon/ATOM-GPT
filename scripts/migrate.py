#!/usr/bin/env python3
"""
Database migration CLI tool for ATOM-GPT
"""
import argparse
import sys
import os
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from database.migrator import DatabaseMigrator
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description='ATOM-GPT Database Migration Tool')
    parser.add_argument('--db-path', default='backend/database/atom_gpt.db', 
                       help='Database file path (default: backend/database/atom_gpt.db)')
    parser.add_argument('command', choices=['migrate', 'status', 'fresh'], 
                       help='Migration command')
    
    args = parser.parse_args()
    
    # Ensure database directory exists
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    migrator = DatabaseMigrator(args.db_path)
    
    if args.command == 'migrate':
        print("🚀 Running migrations...")
        count, executed = migrator.migrate()
        
        if count > 0:
            print(f"✅ Executed {count} migrations:")
            for migration in executed:
                print(f"   📄 {migration}")
        else:
            print("📋 No migrations to execute - database is up to date")
    
    elif args.command == 'status':
        status = migrator.status()
        print(f"📊 Migration Status:")
        print(f"   Database: {status['database_path']}")
        print(f"   Executed: {status['executed_count']} migrations")
        print(f"   Pending: {status['pending_count']} migrations")
        
        if status['executed_migrations']:
            print("\n✅ Executed migrations:")
            for migration in status['executed_migrations']:
                print(f"   📄 {migration}")
        
        if status['pending_migrations']:
            print("\n⏳ Pending migrations:")
            for migration in status['pending_migrations']:
                print(f"   📄 {migration}")
        
        if not status['pending_migrations']:
            print("\n🎉 Database is up to date!")
    
    elif args.command == 'fresh':
        print("⚠️  Fresh migration will delete all data!")
        confirm = input("Are you sure? Type 'yes' to continue: ")
        if confirm.lower() == 'yes':
            # Delete database file
            if os.path.exists(args.db_path):
                os.remove(args.db_path)
                print("🗑️  Deleted existing database")
            
            # Run migrations
            count, executed = migrator.migrate()
            print(f"✅ Fresh database created with {count} migrations")
        else:
            print("❌ Fresh migration cancelled")

if __name__ == '__main__':
    main()