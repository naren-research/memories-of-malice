import sqlite3
import json
import sys
from pathlib import Path

# --- Configuration ---
PROJ_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = Path('./data/kaminski_emails_sorted.jsonl')
DB_FILE = Path('./data/kaminski_emails.sqlite')
TABLE_NAME = 'emails'

# --- Database Schema ---
# All available fields from the JSONL file, with raw_headers flattened.
# Using TEXT for all fields is safest for this kind of data ingestion.
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    body TEXT,
    date TEXT,
    from_email TEXT,
    to_email TEXT,
    subject TEXT,
    cc TEXT,
    bcc TEXT,
    x_from TEXT,
    x_to TEXT,
    x_cc TEXT,
    x_bcc TEXT
);
"""

# --- Indexes ---
# Indexes on columns that are likely to be queried frequently, as specified by the user.
CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {TABLE_NAME} (timestamp);",
    f"CREATE INDEX IF NOT EXISTS idx_date ON {TABLE_NAME} (date);",
    f"CREATE INDEX IF NOT EXISTS idx_from_email ON {TABLE_NAME} (from_email);",
    f"CREATE INDEX IF NOT EXISTS idx_to_email ON {TABLE_NAME} (to_email);",
    f"CREATE INDEX IF NOT EXISTS idx_subject ON {TABLE_NAME} (subject);",
    f"CREATE INDEX IF NOT EXISTS idx_cc ON {TABLE_NAME} (cc);",
    f"CREATE INDEX IF NOT EXISTS idx_bcc ON {TABLE_NAME} (bcc);",
    f"CREATE INDEX IF NOT EXISTS idx_x_from ON {TABLE_NAME} (x_from);",
    f"CREATE INDEX IF NOT EXISTS idx_x_to ON {TABLE_NAME} (x_to);",
    f"CREATE INDEX IF NOT EXISTS idx_x_cc ON {TABLE_NAME} (x_cc);",
    f"CREATE INDEX IF NOT EXISTS idx_x_bcc ON {TABLE_NAME} (x_bcc);"
]

def create_database_and_table(conn):
    """Creates the database table and indexes."""
    try:
        cursor = conn.cursor()
        print(f"Creating table '{TABLE_NAME}'...")
        cursor.execute(CREATE_TABLE_SQL)
        print("Creating indexes...")
        for index_sql in CREATE_INDEXES_SQL:
            cursor.execute(index_sql)
        conn.commit()
        print("Database and table are ready.")
    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)

def load_data(conn):
    """Reads the JSONL file and inserts data into the SQLite table."""
    cursor = conn.cursor()
    insert_sql = f"""INSERT OR REPLACE INTO {TABLE_NAME} (
        id, timestamp, body, date, from_email, to_email, subject, 
        cc, bcc, x_from, x_to, x_cc, x_bcc
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    print(f"Loading data from {SOURCE_FILE}...")
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                email_data = json.loads(line)
                hdr = email_data.get('raw_headers', {}) or {}

                # Prepare a tuple of values in the correct order for insertion
                row_data = (
                    email_data.get('id'),
                    email_data.get('timestamp'),
                    email_data.get('body'),
                    hdr.get('Date'),
                    hdr.get('From'),
                    hdr.get('To'),
                    hdr.get('Subject'),
                    hdr.get('Cc'),
                    hdr.get('Bcc'),
                    hdr.get('X-From'),
                    hdr.get('X-To'),
                    hdr.get('X-cc'),
                    hdr.get('X-bcc')
                )
                cursor.execute(insert_sql, row_data)

                if (i + 1) % 1000 == 0:
                    print(f"... {i + 1} records inserted")
                    conn.commit() # Commit in batches

            conn.commit() # Final commit
            print(f"Finished loading. Total records inserted: {i + 1}")

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing source file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to set up DB and load data."""
    # Create output directory if it doesn't exist
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # If the database file already exists, you might want to delete it
    # to ensure a fresh start. Uncomment the next two lines to do so.
    if DB_FILE.exists():
        print(f"{DB_FILE} exists. Deleting old database file.")
        DB_FILE.unlink()
    else:
        print(f"{DB_FILE} does not exist. Creating new database.")

    try:
        conn = sqlite3.connect(DB_FILE)
        create_database_and_table(conn)
        load_data(conn)
        conn.close()
        print("Process complete.")
    except sqlite3.Error as e:
        print(f"Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
