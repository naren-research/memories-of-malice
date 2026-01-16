#!/usr/bin/env python3
"""
Script to parse Vince Kaminski's Enron emails into JSONL format.
Preserves all raw headers and message content without any filtering or cleaning.
"""

import os
import uuid
import json
import email
import logging
from datetime import datetime
from pathlib import Path
from email.utils import parsedate_to_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ENRON_DIR = Path(__file__).parent.parent.parent / "datasets" / "enron-2015-05-07" / "enron_user_data" / "kaminski-v"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "kaminski_emails.jsonl"

def parse_email_file(file_path: Path) -> dict:
    """Parse a single email file into a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Parse email message
        msg = email.message_from_string(content)
        
        # Get date from headers or fallback to file modification time
        date_str = msg.get('Date')
        if date_str:
            try:
                date = parsedate_to_datetime(date_str)
            except (TypeError, ValueError):
                date = datetime.fromtimestamp(os.path.getmtime(file_path))
        else:
            date = datetime.fromtimestamp(os.path.getmtime(file_path))
            
        # Extract message body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='replace')
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace') if msg.get_payload() else ""

        # Create email record
        email_record = {
            "id": str(uuid.uuid4()),
            "file_path": str(file_path.relative_to(ENRON_DIR)),
            "timestamp": date.isoformat(),
            "raw_headers": dict(msg.items()),  # Preserve all headers
            "body": body
        }
        
        return email_record
        
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
        return None

def main():
    """Main function to process all email files."""
    if not ENRON_DIR.exists():
        logger.error(f"Directory not found: {ENRON_DIR}")
        return

    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    successful_parses = 0
    failed_files = []
    
    # Process all files (skipping directories)
    logger.info(f"Starting to process emails from: {ENRON_DIR}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for file_path in ENRON_DIR.rglob('*'):
            # Skip directories and hidden files
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            try:
                email_record = parse_email_file(file_path)
                if email_record:
                    json.dump(email_record, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    successful_parses += 1
                    if successful_parses % 1000 == 0:
                        logger.info(f"Processed {successful_parses} emails...")
                else:
                    failed_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
                failed_files.append(str(file_path))

    # Print summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully parsed: {successful_parses} emails")
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    
    if failed_files:
        logger.warning(f"\nFailed to parse {len(failed_files)} files:")
        for failed_file in failed_files:
            logger.warning(f"- {failed_file}")

if __name__ == "__main__":
    main()
