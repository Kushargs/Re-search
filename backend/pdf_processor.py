import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import time
import argparse
import concurrent.futures
from functools import partial
import tqdm
import sys
import threading

# Import LangChain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain


# Load environment variables
load_dotenv()

# Configure logging - set file handler to INFO, but console to WARNING to reduce noise
file_handler = logging.FileHandler("pdf_processor.log")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("pdf_processor")

# Disable verbose logging from OpenAI and other libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain.chains").setLevel(logging.WARNING)
logging.getLogger("langchain.llms").setLevel(logging.WARNING)

# Set environment variable to disable LangChain verbose output
os.environ["LANGCHAIN_VERBOSE"] = "false"

# Database connection settings
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "arxiv-postgresDB")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# We'll create LLM instances per worker to avoid sharing across threads
def get_llm():
    """Create a new LLM instance"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False)

# Define main functions
def get_db_session():
    """Get a database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def load_pdf(pdf_path):
    """
    Load a PDF and extract its text content using PyMuPDF
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects containing the PDF content
    """
    logger.info(f"Loading PDF from {pdf_path}")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Successfully loaded PDF with {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks for better processing
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of split Document objects
    """
    logger.info(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        return documents  # Return original documents if splitting fails

def create_stuff_chain(llm):
    """
    Create a summarization chain using the 'stuff' method
    This is suitable for smaller documents that fit within the context window
    
    Returns:
        Chain: LangChain summarization chain
    """
    prompt = ChatPromptTemplate.from_template(
        "Write a concise scientific summary of the following quantum computing paper:\n\n"
        "{text}\n\n"
        "Focus on the key innovations, methodology, and results. "
        "The summary should be detailed enough for a physicist to understand "
        "the main contributions but concise (about 3-4 paragraphs)."
    )
    
    return load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=False
    )

def create_map_reduce_chain(llm):
    """
    Create a summarization chain using the 'map_reduce' method
    This is suitable for longer documents that exceed the context window
    
    Returns:
        Chain: LangChain summarization chain
    """
    # Map prompt - used to summarize each chunk
    map_prompt = ChatPromptTemplate.from_template(
        "Write a scientific summary of this excerpt from a quantum computing paper:\n\n"
        "{text}\n\n"
        "Focus on capturing the technical details, methodologies and results mentioned in this section."
    )
    
    # Combine prompt - used to create a final summary from the individual summaries
    combine_prompt = ChatPromptTemplate.from_template(
        "You are an expert in quantum computing and physics tasked with creating a comprehensive summary.\n\n"
        "Below are summaries of different sections from a quantum computing research paper:\n\n"
        "{text}\n\n"
        "Create a coherent, well-organized summary of the entire paper based on these section summaries.\n"
        "The summary should cover:\n"
        "1. Main objectives and research questions\n"
        "2. Key methodologies and approaches\n"
        "3. Principal findings and results\n"
        "4. Significance and implications for quantum computing\n\n"
        "Your summary should be suitable for a technical audience but concise (3-4 paragraphs)."
    )
    
    return load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False
    )

# Define a wrapper for the map_reduce chain that tracks progress
class ProgressTrackingMapReduceChain:
    def __init__(self, llm, chunks, progress_bar):
        self.llm = llm
        self.chunks = chunks
        self.progress_bar = progress_bar
        
        # Create the map prompt separately instead of trying to access it from the chain
        self.map_prompt = ChatPromptTemplate.from_template(
            "Write a scientific summary of this excerpt from a quantum computing paper:\n\n"
            "{text}\n\n"
            "Focus on capturing the technical details, methodologies and results mentioned in this section."
        )
        
        # Create the combine prompt separately
        self.combine_prompt = ChatPromptTemplate.from_template(
            "You are an expert in quantum computing and physics tasked with creating a comprehensive summary.\n\n"
            "Below are summaries of different sections from a quantum computing research paper:\n\n"
            "{text}\n\n"
            "Create a coherent, well-organized summary of the entire paper based on these section summaries.\n"
            "The summary should cover:\n"
            "1. Main objectives and research questions\n"
            "2. Key methodologies and approaches\n"
            "3. Principal findings and results\n"
            "4. Significance and implications for quantum computing\n\n"
            "Your summary should be suitable for a technical audience but concise (3-4 paragraphs)."
        )
        
    def invoke(self, inputs):
        """
        Process the documents while updating the progress bar
        """
        # First phase: Process each chunk individually (map phase)
        self.progress_bar.set_description("Mapping chunks")
        
        # Create a copy of inputs to avoid modifying the original
        modified_inputs = dict(inputs)
        
        # We'll implement our own map-reduce logic here
        map_results = []
        
        # Process each chunk with individual progress updates
        for i, chunk in enumerate(self.chunks):
            # Update description with current chunk
            self.progress_bar.set_description(f"Mapping chunk {i+1}/{len(self.chunks)}")
            
            # Process current chunk - we'll use stuff chain for each chunk
            single_chunk_chain = load_summarize_chain(
                self.llm, 
                chain_type="stuff",
                prompt=self.map_prompt,
                verbose=False
            )
            
            result = single_chunk_chain.invoke({"input_documents": [chunk]})
            map_results.append(result["output_text"])
            
            # Update progress for this chunk
            self.progress_bar.update(1)
        
        # Second phase: Combine the summaries (reduce phase)
        # Update the progress bar for the combine phase
        self.progress_bar.set_description("Combining summaries")
        
        # Create a new document from each map result
        from langchain_core.documents import Document
        combine_docs = [Document(page_content=text) for text in map_results]
        
        # Run the combine step
        combine_chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=self.combine_prompt,
            verbose=False
        )
        
        final_result = combine_chain.invoke({"input_documents": combine_docs})
        
        # Final update to progress bar
        self.progress_bar.set_description("Completed")
        
        return final_result
    
def update_paper_with_summary(paper_id, summary):
    """
    Update the paper record in the database with the generated summary
    
    Args:
        paper_id (str): arXiv ID of the paper
        summary (str): Generated summary of the paper
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    session = get_db_session()
    try:
        query = text("""
            UPDATE papers
            SET 
                pdf_processed = TRUE,
                pdf_process_date = :process_date,
                ai_summary = :summary
            WHERE id = :paper_id
        """)
        
        session.execute(query, {
            "process_date": datetime.now(),
            "summary": summary,
            "paper_id": paper_id
        })
        
        session.commit()
        logger.info(f"Updated database with summary for paper {paper_id}")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating paper summary for {paper_id}: {e}")
        return False
    finally:
        session.close()

def get_papers_for_processing(limit=5, category=None):
    """
    Get a list of papers that have PDFs but haven't been processed yet
    
    Args:
        limit (int): Maximum number of papers to process
        category (str, optional): Specific category to filter by (e.g., 'quant-ph')
        
    Returns:
        list: List of paper records to process
    """
    session = get_db_session()
    try:
        # Base query
        query_str = """
            SELECT id, pdf_path 
            FROM papers 
            WHERE pdf_fetched = TRUE 
            AND pdf_processed = FALSE
        """
        
        # Add category filter if provided
        if category:
            query_str += f" AND '{category}' = ANY(categories)"
        
        # Add limit
        query_str += " ORDER BY created DESC LIMIT :limit"
        
        # Execute query
        result = session.execute(text(query_str), {"limit": limit})
        papers = result.fetchall()
        
        logger.info(f"Found {len(papers)} papers to process")
        return papers
        
    except Exception as e:
        logger.error(f"Error fetching papers for processing: {e}")
        return []
    finally:
        session.close()
    
def process_paper(paper_data, position=0):
    """
    Process a single paper: load PDF, split text, generate summary with nested progress tracking
    
    Args:
        paper_data: A tuple containing (paper_id, pdf_path)
        position: Position for the nested progress bar
        
    Returns:
        tuple: (paper_id, success, summary) - ID, whether processing was successful, and the generated summary
    """
    paper_id, pdf_path = paper_data
    
    try:
        # Create a new LLM instance for this worker
        llm = get_llm()
        
        # Load the PDF
        documents = load_pdf(pdf_path)
        if not documents:
            logger.error(f"Failed to load PDF for paper {paper_id}")
            return paper_id, False, None
            
        # Determine which summarization approach to use based on document length
        total_text_length = sum(len(doc.page_content) for doc in documents)
        logger.info(f"Paper {paper_id} has {total_text_length} characters across {len(documents)} pages")
        
        # Split into chunks
        chunks = split_documents(documents)
        
        # Create a nested progress bar for this paper
        # For stuff method, we only have 2 steps: processing and complete
        # For map_reduce, we track each chunk plus the combine step
        if total_text_length < 4000:  # Stuff method
            total_steps = 2
            subprogress_desc = f"Paper {paper_id} (stuff method)"
        else:  # Map-reduce method
            total_steps = len(chunks) + 1  # +1 for the final combine step
            subprogress_desc = f"Paper {paper_id} (map-reduce)"
            
        # Create the nested progress bar
        with tqdm.tqdm(
            total=total_steps,
            desc=subprogress_desc,
            position=position,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}"
        ) as subprogress:
            # Choose summarization strategy based on length
            try:
                if total_text_length < 4000:  # For shorter papers
                    logger.info(f"Using 'stuff' method for paper {paper_id}")
                    subprogress.set_description("Processing with stuff method")
                    chain = create_stuff_chain(llm)
                    summary = chain.invoke({"input_documents": documents})
                    subprogress.update(2)  # Complete both steps at once for stuff method
                else:  # For longer papers
                    logger.info(f"Using 'map_reduce' method for paper {paper_id}")
                    
                    # Use our progress tracking wrapper
                    tracking_chain = ProgressTrackingMapReduceChain(llm, chunks, subprogress)
                    summary = tracking_chain.invoke({"input_documents": chunks})
                    
                    # Make sure progress is complete
                    if subprogress.n < subprogress.total:
                        subprogress.update(subprogress.total - subprogress.n)
            except Exception as e:
                logger.error(f"Error in summarization for paper {paper_id}: {e}")
                subprogress.set_description(f"Failed: {str(e)[:30]}...")
                raise e
        
        # Extract the summary text from the response
        if isinstance(summary, dict) and "output_text" in summary:
            final_summary = summary["output_text"]
        else:
            final_summary = str(summary)
            
        logger.info(f"Successfully generated summary for paper {paper_id}")
        return paper_id, True, final_summary
        
    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
        return paper_id, False, None

def process_pending_papers_parallel(limit=5, category="quant-ph", max_workers=4):
    """
    Process a batch of papers in parallel that need summarization
    
    Args:
        limit (int): Maximum number of papers to process
        category (str): Category to filter by (e.g., 'quant-ph')
        max_workers (int): Maximum number of concurrent workers
        
    Returns:
        int: Number of successfully processed papers
    """
    success_count = 0
    
    try:
        # Get papers that need processing
        papers = get_papers_for_processing(limit, category)
        if not papers:
            print("No papers found for processing")
            return 0
            
        total_papers = len(papers)
        print(f"Starting processing of {total_papers} papers with {max_workers} parallel workers")
        
        # Create the main progress bar for overall progress
        with tqdm.tqdm(
            total=total_papers,
            desc="Overall progress",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}",
            leave=True
        ) as main_progress:
            
            # Dictionary to track paper status
            paper_statuses = {paper.id: "pending" for paper in papers}
            
            # Process papers in parallel
            # For parallel processing with nested bars, we need to be careful with positioning
            # Process at most max_workers papers at a time
            for batch_idx in range(0, len(papers), max_workers):
                batch = papers[batch_idx:min(batch_idx + max_workers, len(papers))]
                futures = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit batch of papers for processing
                    for i, paper in enumerate(batch):
                        # Position is i+1 since position 0 is used by the main progress bar
                        future = executor.submit(process_paper, paper, position=i+1)
                        futures.append((future, paper))
                        
                    # Process results as they complete
                    for future, paper in futures:
                        try:
                            paper_id, success, summary = future.result()
                            
                            # Update paper status
                            if success and summary:
                                if update_paper_with_summary(paper_id, summary):
                                    success_count += 1
                                    paper_statuses[paper_id] = "success"
                                    main_progress.set_description(f"Success: {success_count}/{total_papers}")
                                else:
                                    paper_statuses[paper_id] = "db_error"
                                    main_progress.set_description(f"DB Error: {paper_id}")
                            else:
                                paper_statuses[paper_id] = "process_error"
                                main_progress.set_description(f"Process Error: {paper_id}")
                                
                            # Update the main progress bar
                            main_progress.update(1)
                                
                        except Exception as e:
                            logger.error(f"Error processing future: {e}")
                            main_progress.update(1)
                
                # Print a newline between batches to keep progress bars separated
                if batch_idx + max_workers < len(papers):
                    print("\n" * (max_workers + 1))
        
        # Print final summary
        print(f"\nCompleted parallel processing:")
        print(f"- Successfully processed: {success_count}/{total_papers}")
        print(f"- Failed: {total_papers - success_count}/{total_papers}")
        
        # Log detailed status
        logger.info(f"Completed parallel processing. Successfully processed {success_count}/{total_papers} papers")
        return success_count
        
    except Exception as e:
        logger.error(f"Error in process_pending_papers_parallel: {e}", exc_info=True)
        return success_count

def process_single_paper(paper_id):
    """
    Process a single paper by ID with detailed progress tracking
    
    Args:
        paper_id (str): ID of the paper to process
        
    Returns:
        bool: Whether processing was successful
    """
    session = get_db_session()
    try:
        result = session.execute(
            text("SELECT id, pdf_path FROM papers WHERE id = :id"),
            {"id": paper_id}
        ).fetchone()
        
        if not result:
            print(f"Paper not found: {paper_id}")
            return False
            
        # Process the paper with detailed progress tracking (position 0 since it's the only one)
        paper_id, success, summary = process_paper((result.id, result.pdf_path), position=0)
        
        if success and summary:
            if update_paper_with_summary(paper_id, summary):
                print(f"✅ Successfully processed paper {paper_id}")
                return True
            else:
                print(f"❌ Failed to update database for paper {paper_id}")
                return False
        else:
            print(f"❌ Failed to process paper {paper_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
        print(f"❌ Error processing paper {paper_id}: {str(e)}")
        return False
    finally:
        session.close()

def main():
    """
    Main function to run the PDF processor
    """
    parser = argparse.ArgumentParser(description="Process arXiv PDFs and generate summaries")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of papers to process")
    parser.add_argument("--category", type=str, default="quant-ph", 
                        help="Category to filter by (e.g., 'quant-ph', 'cs.QC')")
    parser.add_argument("--paper-id", type=str, help="Process a specific paper by ID")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose console output")
    args = parser.parse_args()
    
    # Enable verbose logging if requested
    if args.verbose:
        console_handler.setLevel(logging.INFO)
        print("Verbose logging enabled")
    
    logger.info("Starting PDF processor")
    print("Starting PDF processor...")
    
    start_time = time.time()
    
    if args.paper_id:
        # Process a specific paper
        print(f"Processing specific paper: {args.paper_id}")
        process_single_paper(args.paper_id)
    else:
        # Process a batch of papers in parallel
        print(f"Processing batch of up to {args.limit} papers in category '{args.category}' with {args.workers} workers")
        print("Each paper will show individual progress for chunks processing\n")
        processed = process_pending_papers_parallel(args.limit, args.category, args.workers)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format elapsed time nicely
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds:.2f} seconds"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        time_str = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    
    print(f"\nPDF processor completed in {time_str}")
    logger.info(f"PDF processor completed in {time_str}")

if __name__ == "__main__":
    main()