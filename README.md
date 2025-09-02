# Email Triage with LangChain and Ollama

## Project Overview
This is a proof-of-concept (PoC) for an intelligent email triage system. It demonstrates how to use a local Large Language Model (LLM), specifically **Llama3 via Ollama**, to automate the process of handling incoming emails.  

The system automatically **summarizes, classifies, and takes a logical, automated action** based on the email's content.  

The primary goal is to showcase a practical, end-to-end **LangChain pipeline** that can be run entirely on a local machine, without relying on external APIs.  

---

## Key Features
- **Automated Summarization**  
  Condenses the content of an email into a concise, three-sentence summary.  

- **Smart Classification**  
  Categorizes emails into *Urgent*, *Important*, or *General* using a robust classification chain.  

- **Intelligent Agent**  
  Utilizes a ReAct agent to decide which tool to use (e.g., send a Slack notification or add a task) based on the email's classification.  

- **Actionable Tools**  
  Includes custom-built tools that simulate real-world actions like sending Slack notifications, adding to-do list items, and archiving emails.  

- **Robust Parsing**  
  Employs a Pydantic-based `JsonOutputParser` to ensure reliable communication with the LLM and prevent common JSON parsing errors.  

---

## Architecture
The system is built as a **LangChain agentic loop**. The process for each email is as follows:  

1. **Summarization Chain**  
   The raw email content is passed to a summarization chain, which generates a short summary.  

2. **Classification Chain**  
   The summary is then passed to a classification chain. This chain, with the help of a `JsonOutputParser`, prompts the LLM to output a JSON object containing the email's category and a reason.  

3. **Agent Executor**  
   Based on the classified category, a prompt is crafted and sent to the `AgentExecutor`.  

4. **ReAct Agent**  
   The agent uses a ReAct prompt template to decide which of the available tools to use to address the email. It determines the correct tool and its input, then executes the action.  

---

## Prerequisites
Before running the application, you need to have the following installed:  

- **Python 3.9+**  
- **Ollama**: A local LLM server.  
  - Download and install Ollama from [https://ollama.ai/](https://ollama.ai/).  
  - Pull the Llama3 model:  
    ```bash
    ollama pull llama3
    ```  
  - Ensure the Ollama server is running in the background.  

---

## Setup and Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/snehalathag-infomagnus/langchain-poc.git
   cd langchain-poc
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**
    Create a .env file in the root directory and add any necessary API keys.
    (This PoC does not require them for core functionality, but it's good practice for future expansion.)


**Usage**
Simply run the main Python script from your terminal:

```python
python main.py
```

The script will process the sample emails, and you will see the output for each step: summarization, classification, and the agent's thought process as it decides and executes an action.