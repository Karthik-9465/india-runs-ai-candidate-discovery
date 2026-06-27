# src/utils.py
import json
import logging
import sys
from datetime import datetime

def setup_logging():
    """Configure robust, clean logging for the console."""
    logger = logging.getLogger("CandidateDiscovery")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def parse_date(date_str):
    """Parse a date string in YYYY-MM-DD format safely."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None

def calculate_months_between(start_date_str, end_date_str, current_date):
    """Calculate the duration in months between two dates, using current_date if end is null."""
    start_dt = parse_date(start_date_str)
    if not start_dt:
        return 0
    
    if end_date_str:
        end_dt = parse_date(end_date_str)
    else:
        end_dt = current_date
        
    if not end_dt:
        return 0
        
    return (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

def load_candidates_generator(file_path):
    """Memory-efficient generator to load candidates one-by-one from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logging.getLogger("CandidateDiscovery").warning(f"Failed to parse line as JSON: {e}")
                continue

def save_json(data, file_path):
    """Save data to a JSON file with UTF-8 encoding."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path):
    """Load data from a JSON file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
