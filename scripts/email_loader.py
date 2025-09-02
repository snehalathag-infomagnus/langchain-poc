import os
from langchain_core.documents import Document

def load_sample_emails():
    """
    Simulates loading emails for the PoC.
    In a real application, you would use imaplib or a dedicated email API client.
    """
    sample_emails = [
        """
        Subject: Server Downtime
        
        Hi Team,
        
        Our main production server is currently experiencing an outage. We need all hands on deck to investigate this immediately. Please check the incident channel in Teams for updates. We are working on a resolution.
        
        - John
        """,
        """
        Subject: Project Status Update
        
        Hi All,
        
        This is a reminder that the Q3 project status report is due by end of business tomorrow. Please submit your individual updates to the shared drive.
        
        Thanks,
        Jane
        """,
        """
        Subject: Weekend plans?
        
        Hey,
        
        Hope you're having a good week. What are your plans for the weekend? Thinking of checking out that new movie.
        
        - Alex
        """
    ]
    
    # Convert text to LangChain Document objects
    email_docs = []
    for email_text in sample_emails:
        subject = email_text.strip().split('\n')[0].replace('Subject: ', '')
        doc = Document(page_content=email_text, metadata={"subject": subject})
        email_docs.append(doc)
    
    return email_docs