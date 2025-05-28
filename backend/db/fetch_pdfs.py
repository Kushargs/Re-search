from datetime import datetime
import logging
import os
import time
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import sys
import argparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_pdfs.log"),
        logging.StreamHandler()
    ]
)

# Database connection settings
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "arxiv-postgresDB")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# PDF storage directory
PDF_DIR = os.getenv("PDF_STORAGE_PATH", os.path.join(os.getcwd(), "pdfs"))
os.makedirs(PDF_DIR, exist_ok=True)

# ArXiv API settings
USER_AGENT = os.getenv("ARXIV_USER_AGENT", "arxiv-fetcher/0.1 (mailto:your@email.com)")
FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "10"))  # Process 10 papers at a time
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "3.0"))  # Seconds between requests
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Category definitions
CATEGORY_GROUPS = {
    "quantum": ["quant-ph", "cs.QC"],  # Quantum computing and quantum physics
    # "ai": ["cs.AI", "cs.LG", "cs.CL", "stat.ML"],  # AI and machine learning
    # "physics": ["physics", "cond-mat", "hep-th", "astro-ph"],  # General physics
    # "math": ["math"],  # Mathematics
    # "biology": ["q-bio"],  # Quantitative biology
    # Add more category groups as needed
}

def get_db_session(max_retries=3):
    """Establish database connection with retry logic"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            engine = create_engine(DATABASE_URL)
            Session = sessionmaker(bind=engine)
            return Session()
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
            wait_time = 5 * retry_count
            logging.warning(f"Database connection error. Retrying in {wait_time}s ({retry_count}/{max_retries})")
            time.sleep(wait_time)

def build_category_filter(categories):
    """
    Build SQL filter condition for the given categories.
    
    Args:
        categories: List of category codes or a category group name
        
    Returns:
        SQL condition for the WHERE clause
    """
    # If categories is a string, check if it's a category group
    if isinstance(categories, str):
        if categories in CATEGORY_GROUPS:
            categories = CATEGORY_GROUPS[categories]
        else:
            # Treat as a single category
            categories = [categories]
    
    # Build the filter condition
    if not categories:
        return "TRUE"  # No filtering if empty list
        
    # Create an OR condition for each category
    category_conditions = []
    for category in categories:
        category_conditions.append(f"'{category}' = ANY(categories)")
    
    return "(" + " OR ".join(category_conditions) + ")"

def get_papers_needing_pdfs(session, categories=None, limit=FETCH_LIMIT):
    """
    Get papers that don't have PDFs yet, optionally filtered by categories.
    
    Args:
        session: Database session
        categories: List of categories or a category group name (e.g., "quantum")
        limit: Maximum number of papers to fetch
        
    Returns:
        List of papers needing PDFs
    """
    try:
        # Build category filter
        category_filter = build_category_filter(categories)
        
        # SQL query to fetch papers
        query = text(f"""
            SELECT id, link, pdf_fetch_attempts
            FROM papers
            WHERE {category_filter}
            AND pdf_fetched = FALSE
            AND pdf_fetch_attempts < :max_retries
            ORDER BY pdf_fetch_attempts ASC, created DESC
            LIMIT :limit
        """)
        
        result = session.execute(query, {"max_retries": MAX_RETRIES, "limit": limit})
        papers = [{"id": row.id, "link": row.link, "attempts": row.pdf_fetch_attempts} for row in result]
        
        category_desc = categories if categories else "all categories"
        logging.info(f"Found {len(papers)} papers from {category_desc} needing PDFs")
        return papers
        
    except Exception as e:
        logging.error(f"Error fetching papers from database: {e}", exc_info=True)
        return []

def download_pdf(paper_id, paper_link):
    """
    Download PDF for a paper and save it to the PDF directory.
    Returns the path to the saved PDF or None if download failed.
    """
    # Convert link from /abs/ format to /pdf/ format
    if '/abs/' in paper_link:
        pdf_link = paper_link.replace('/abs/', '/pdf/') + '.pdf'
    else:
        pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
    
    # Check if PDF already exists
    if os.path.exists(pdf_path):
        logging.info(f"PDF already exists for {paper_id}")
        return pdf_path
    
    try:
        logging.info(f"Downloading PDF for {paper_id} from {pdf_link}")
        
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf"
        }
        
        response = requests.get(pdf_link, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            # Save the PDF
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Verify the file was saved and is not empty
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                logging.info(f"Successfully downloaded PDF for {paper_id} ({os.path.getsize(pdf_path)} bytes)")
                return pdf_path
            else:
                logging.error(f"PDF file for {paper_id} is empty or not saved properly")
                return None
        else:
            logging.error(f"Failed to download PDF for {paper_id}. Status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading PDF for {paper_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading PDF for {paper_id}: {e}")
        return None

def update_paper_status(session, paper_id, success, pdf_path=None):
    """
    Update the paper's PDF status in the database.
    """
    try:
        query = text("""
            UPDATE papers
            SET 
                pdf_fetched = :success,
                pdf_fetch_attempts = pdf_fetch_attempts + 1,
                pdf_fetch_date = :fetch_date,
                pdf_path = :pdf_path
            WHERE id = :paper_id
        """)
        
        session.execute(query, {
            "success": success,
            "fetch_date": datetime.now(),
            "pdf_path": pdf_path,
            "paper_id": paper_id
        })
        
        session.commit()
        logging.info(f"Updated database status for {paper_id}: pdf_fetched={success}")
        
    except Exception as e:
        session.rollback()
        logging.error(f"Error updating paper status for {paper_id}: {e}")

def fetch_pdfs(categories=None, limit=None):
    """
    Main function to fetch PDFs for papers in specified categories.
    
    Args:
        categories: List of categories or a category group name (e.g., "quantum")
        limit: Maximum number of papers to process (overrides FETCH_LIMIT)
    """
    try:
        session = get_db_session()
        
        # Use provided limit or default
        fetch_limit = limit if limit is not None else FETCH_LIMIT
        
        # Get papers needing PDFs
        papers = get_papers_needing_pdfs(session, categories, fetch_limit)
        
        if not papers:
            category_desc = categories if categories else "any category"
            logging.info(f"No papers from {category_desc} need PDF fetching")
            return
        
        # Stats for reporting
        stats = {"success": 0, "failed": 0}
        
        # Process each paper
        for i, paper in enumerate(papers):
            paper_id = paper["id"]
            paper_link = paper["link"]
            
            logging.info(f"Processing paper {i+1}/{len(papers)}: {paper_id}")
            
            # Download the PDF
            pdf_path = download_pdf(paper_id, paper_link)
            
            # Update database
            if pdf_path:
                update_paper_status(session, paper_id, True, pdf_path)
                stats["success"] += 1
            else:
                update_paper_status(session, paper_id, False, None)
                stats["failed"] += 1
            
            # Rate limiting - be nice to arXiv
            if i < len(papers) - 1:  # No need to wait after the last paper
                logging.debug(f"Waiting {MIN_REQUEST_INTERVAL} seconds before next request")
                time.sleep(MIN_REQUEST_INTERVAL)
        
        # Log summary
        category_desc = categories if categories else "all categories"
        logging.info(f"PDF fetching completed for {category_desc}. Results: {stats}")
        
    except Exception as e:
        logging.error(f"Error in fetch_pdfs: {e}", exc_info=True)
    finally:
        if 'session' in locals():
            session.close()

def list_categories():
    """List all available category groups and their categories"""
    print("Available category groups:")
    for group, categories in CATEGORY_GROUPS.items():
        print(f"  {group}: {', '.join(categories)}")

def main():
    """
    Main entry point with command-line argument support.
    """
    parser = argparse.ArgumentParser(description="Fetch arXiv PDFs for specified categories")
    parser.add_argument("--category", "-c", help="Category or category group to fetch (default: quantum)")
    parser.add_argument("--limit", "-l", type=int, help=f"Maximum papers to fetch (default: {FETCH_LIMIT})")
    parser.add_argument("--list", action="store_true", help="List available category groups")
    
    args = parser.parse_args()
    
    if args.list:
        list_categories()
        return 0
        
    try:
        category = args.category if args.category else "quantum"
        logging.info(f"Starting PDF fetcher for category: {category}")
        fetch_pdfs(category, args.limit)
        logging.info("PDF fetching process completed")
        return 0
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        return 130
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())