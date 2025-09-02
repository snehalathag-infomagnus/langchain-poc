from dotenv import load_dotenv

# LangChain components
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Local scripts
from scripts.email_loader import load_sample_emails

# --- Step 1: Initial Setup ---
load_dotenv()
llm = OllamaLLM(model="llama3")

# --- Step 2: Custom Tools ---
@tool
def send_slack_notification(message: str) -> str:
    """Sends an urgent message to a designated Slack channel."""
    print(f"\n[Tool: send_slack_notification] Sending Slack notification: {message}")
    return "Slack notification sent successfully."

@tool
def add_to_todo_list(task_description: str) -> str:
    """Adds a new task to your to-do list."""
    print(f"\n[Tool: add_to_todo_list] Adding to-do item: {task_description}")
    return "Task added to to-do list."

@tool
def archive_email() -> str:
    """Archives the current email."""
    print("\n[Tool: archive_email] Archiving email.")
    return "Email archived."

tools = [send_slack_notification, add_to_todo_list, archive_email]

# --- Step 3: Define the Chains ---
print("Defining LLM chains...")

# Summarization Chain
summary_template = """
You are an expert at summarizing emails. Summarize the following email in a short paragraph, no more than three sentences.
Email content: {email_content}
Summary:
"""
summary_prompt = PromptTemplate.from_template(summary_template)
summarization_chain = (
    {"email_content": RunnablePassthrough()}
    | summary_prompt
    | llm
)

# Classification Chain with Pydantic for reliable JSON parsing
class EmailClassification(BaseModel):
    category: str = Field(description="The classification of the email (Urgent, Important, or General).")
    reason: str = Field(description="A brief reason for the classification.")

classification_parser = JsonOutputParser(pydantic_object=EmailClassification)
classification_template = """
You are an intelligent email assistant.
Given the email summary, classify it as 'Urgent', 'Important', or 'General'.
Provide a brief reason for your classification.
{format_instructions}
Email Summary: {summary}

IMPORTANT: Output ONLY valid JSON. Do not include any extra text, explanation, or formatting.
"""
classification_prompt = PromptTemplate.from_template(
    classification_template,
    partial_variables={"format_instructions": classification_parser.get_format_instructions()},
)

classification_chain = (
    {"summary": RunnablePassthrough()}
    | classification_prompt
    | llm
    | classification_parser
)

# --- Step 4: Create the Agent ---
print("Creating the LangChain agent...")
agent_prompt = hub.pull("hwchase17/react-json")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Step 5: Main Orchestration Loop ---
if __name__ == "__main__":
    print("\n--- Starting Email Triage PoC ---")
    emails = load_sample_emails()
    
    for i, email_doc in enumerate(emails):
        print(f"\n\n--- Processing Email {i+1}/{len(emails)}: {email_doc.metadata['subject']} ---")
        
        # 1. Summarize the email
        summary_result = summarization_chain.invoke(email_doc.page_content)
        summary = summary_result.strip()
        print(f"Summary: {summary}")
        
        # 2. Classify the summary using the robust chain
        classification_data = classification_chain.invoke({"summary": summary})
        category = classification_data.get("category", "General")
        reason = classification_data.get("reason", "Failed to classify automatically.")
        
        print(f"Classification: {category} (Reason: {reason})")
        
        # 3. Let the agent take action based on the classification
        agent_instruction = (
            "IMPORTANT: When you take an action, use the following format exactly:\n"
            "Action: <tool name>\n"
            "Action Input: <input for the tool>\n"
            "Do not use JSON or code blocks."
        )
        if category == 'Urgent':
            agent_input = f"This email is urgent. The summary is: '{summary}'. It requires immediate attention. Use the appropriate tool to notify me. {agent_instruction}"
        elif category == 'Important':
            agent_input = f"This email is important. The summary is: '{summary}'. It contains a task that needs to be done. Use the appropriate tool to track this task. {agent_instruction}"
        elif category == 'General':
            agent_input = f"This is a general email. The summary is: '{summary}'. It does not require a specific action. Use the appropriate tool to handle it. {agent_instruction}"
        else:
            agent_input = f"The classification was unclear. Summary: '{summary}'. Use the archive tool as a default. {agent_instruction}"
            
        print(f"\nAgent's thought process starts...")
        agent_executor.invoke({"input": agent_input})