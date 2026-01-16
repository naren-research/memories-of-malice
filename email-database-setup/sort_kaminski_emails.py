import json
from datetime import datetime

INPUT_PATH = "../datasets/enron-2015-05-07/enron_user_data_parsed/kaminski_emails.jsonl"
OUTPUT_PATH = "../datasets/enron-2015-05-07/enron_user_data_parsed/kaminski_emails_sorted.jsonl"

def load_emails(path):
    emails = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    email = json.loads(line)
                    emails.append(email)
                except json.JSONDecodeError:
                    continue
    return emails

def sort_emails_by_timestamp(emails):
    def get_timestamp(email):
        return email.get('timestamp', '')
    return sorted(emails, key=get_timestamp)

def save_emails(emails, path):
    with open(path, 'w', encoding='utf-8') as f:
        for email in emails:
            f.write(json.dumps(email, ensure_ascii=False) + '\n')

def main():
    emails = load_emails(INPUT_PATH)
    sorted_emails = sort_emails_by_timestamp(emails)
    save_emails(sorted_emails, OUTPUT_PATH)
    print(f"Sorted {len(sorted_emails)} emails by timestamp and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
