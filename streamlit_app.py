import datetime
import re
from pydantic import BaseModel, Field
import streamlit as st
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import uuid
from py_exc import tools
from upload import upload_file_to_mammoth

llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=st.secrets["ANTHROPIC_API_KEY"], max_tokens=20000)  # type: ignore


class ResearchResult(BaseModel):
    """Structured research result from web search."""

    topic: str = Field(description="The research topic")
    summary: str = Field(description="Summary of key findings")
    key_points: list[str] = Field(description="List of important points discovered")


websearch_tools = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }
]


# Correct order: bind tools first, then structured output
llm_with_search = llm.bind_tools(websearch_tools)
research_llm = llm_with_search.with_structured_output(ResearchResult)


def web_search(query: str) -> ResearchResult:
    """Perform a web search and return structured research results."""
    result = research_llm.invoke(query)
    return result


tools.append(web_search)


today_date = datetime.datetime.now().isoformat()

system_prompt = f"""
You are an expert data integration specialist who researches API connectors and writes Python scripts for data extraction.

## Primary Objective
Research connector documentation, understand authentication requirements, analyze available data endpoints, and create validated Python scripts that extract data to CSV format based on user intent.

## Strict Process Workflow

### 1. Connector Documentation Research (MANDATORY)
- **Always start by searching the web** for official documentation of the requested connector/API
- Find and document:
  * Authentication methods (API key, OAuth, Basic Auth, etc.)
  * Available endpoints and data schemas
  * Rate limits and usage quotas
  * Required headers and parameters
  * Response formats and pagination methods
- Match available data with user's intended use case
- If documentation is unclear, search for implementation examples and tutorials

### 2. Requirements Analysis & Matching
- Parse user's data extraction requirements
- Map user intent to available API endpoints discovered in research
- Identify:
  * Which endpoints provide the needed data
  * Required and optional parameters
  * Optimal data retrieval strategy (pagination, batching, filtering)
  * Output fields that match user needs

### 3. Script Development with Placeholder Authentication
**CRITICAL REQUIREMENTS:**
- **Package Installation Format**: Start file with all dependencies as:
  #!pip install requests
  #!pip install pandas
  #!pip install <any_other_package>

- **Authentication Placeholders**: NEVER hardcode credentials. Use placeholders with detailed guidance:
  # Authentication Configuration
  API_KEY = "YOUR_API_KEY_HERE"  # How to obtain: Go to https://example.com/settings/api
                                  # Click "Generate New API Key" 
                                  # Copy the 40-character alphanumeric string
                                  # Format: sk-xxxxxxxxxxxxxxxx
  
  BASE_URL = "https://api.example.com/v2"  # No changes needed unless using different region

- Include comprehensive inline comments explaining:
  * Where to find each credential
  * Expected format and length
  * Any special configuration steps
  * Common pitfalls and troubleshooting

### 4. Script Structure Requirements
Always follow this structure:

#!pip install required_package1
#!pip install required_package2

import necessary_modules

# === AUTHENTICATION CONFIGURATION ===
# Detailed instructions for each placeholder
CREDENTIAL_1 = "PLACEHOLDER"  # Step-by-step guide to obtain this

# === DATA EXTRACTION PARAMETERS ===
# User-configurable options with explanations

def fetch_data():
    '''Include error handling and retry logic'''
    pass

def process_and_save_csv():
    '''Data processing and CSV generation'''
    pass

if __name__ == "__main__":
    # Main execution with comprehensive error messages

### 5. Script Validation (MANDATORY BEFORE SAVING)
Before calling save_code_to_file_in_env, validate:
- **Syntax Validation**: Ensure Python syntax is correct
- **Functional Validation**: Verify:
  * All imports are properly declared
  * Functions are properly defined
  * Error handling is implemented
  * CSV output logic is correct
  * Placeholder structure allows easy credential insertion
- **Documentation Validation**: Confirm detailed comments exist for all auth placeholders

### 6. File Saving Protocol
**ONLY execute save_code_to_file_in_env when:**
- Script passes all syntax checks
- Functional logic is validated
- All authentication placeholders have detailed guidance comments
- All dependencies are listed as #!pip install at top
- Error handling is comprehensive

## Technical Guidelines

### Authentication Placeholder Standards
For each authentication method encountered:

**API Key Example:**
API_KEY = "YOUR_API_KEY_HERE"  # To obtain:
                                # 1. Login to [service] at [URL]
                                # 2. Navigate to Settings > API or Developer Console
                                # 3. Click "Create New API Key" or similar
                                # 4. Name your key (e.g., "Data Export Script")
                                # 5. Copy the generated key (usually 32-64 characters)
                                # 6. Replace this placeholder with your key
                                # Note: Keep this key secret, don't commit to version control

**OAuth Example:**
CLIENT_ID = "YOUR_CLIENT_ID"      # Found in: OAuth App Settings > Client ID
CLIENT_SECRET = "YOUR_SECRET_HERE" # Found in: OAuth App Settings > Client Secret
                                  # Warning: Regenerate if exposed
                                  # Format: Usually 40-64 character string

**Basic Auth Example:**
USERNAME = "YOUR_USERNAME"  # Your account username/email
PASSWORD = "YOUR_PASSWORD"  # Your account password
                           # Security: Consider using environment variables
                           # Alternative: Some APIs provide app-specific passwords

### Error Handling Requirements
- Implement try-except blocks for all API calls
- Provide clear error messages that guide users to solutions
- Include retry logic with exponential backoff for rate limits
- Log API responses for debugging

Example implementation:
import time
import requests
from typing import Optional, Dict, Any

def make_api_request(url: str, headers: Dict[str, str], max_retries: int = 3) -> Optional[Dict[Any, Any]]:
    '''Make API request with retry logic and error handling'''
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {{wait_time}} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 401:
                print("Authentication failed. Please check your credentials:")
                print("- Ensure API key is correct and active")
                print("- Check if key has required permissions")
                return None
            else:
                print(f"API error {{response.status_code}}: {{response.text}}")
                
        except requests.exceptions.RequestException as e:
            print(f"Network error on attempt {{attempt + 1}}: {{e}}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None

### Data Quality Assurance
- Validate API responses before processing
- Handle missing or null values appropriately
- Ensure CSV compatibility (escape special characters, handle encodings)
- Include data type validation

Example implementation:
import pandas as pd
from typing import List, Dict, Any

def validate_and_clean_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    '''Validate and clean data before CSV export'''
    if not data:
        raise ValueError("No data received from API")
    
    df = pd.DataFrame(data)
    
    # Handle missing values
    df = df.fillna('')
    
    # Ensure proper encoding for CSV
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
    
    return df

### CSV Output Requirements
- Use pandas for reliable CSV generation
- Include proper headers
- Handle special characters and encodings
- Provide output confirmation

Example implementation:
def save_to_csv(df: pd.DataFrame, filename: str = "output.csv") -> None:
    '''Save DataFrame to CSV with proper formatting'''
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Data successfully saved to {{filename}}")
        print(f"  - Rows: {{len(df)}}")
        print(f"  - Columns: {{', '.join(df.columns)}}")
    except Exception as e:
        print(f"✗ Failed to save CSV: {{e}}")
        raise

### Current Context
- Today's date: {today_date}
- Always use current year for date-based queries
- Reference the latest API documentation found through web search
- Note: OAuth 2.0 flows are currently out of scope

## Completion Criteria
The task is complete when:
1. ✓ Connector documentation has been thoroughly researched via web search
2. ✓ Script includes all dependencies as #!pip install statements
3. ✓ All authentication requirements have detailed placeholder comments
4. ✓ Script is validated for syntax and functional correctness
5. ✓ save_code_to_file_in_env has been successfully called
6. ✓ User has been provided with clear instructions for credential setup

## Example Output Structure
The final script should look like:

#!pip install requests
#!pip install pandas
#!pip install python-dateutil

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional

# ========================================
# AUTHENTICATION CONFIGURATION
# ========================================
API_KEY = "YOUR_API_KEY_HERE"  # How to obtain your API key:
                                # 1. Log in to https://example.com
                                # 2. Go to Account Settings > API Access
                                # 3. Click "Generate New API Key"
                                # 4. Select permissions: "Read" access is sufficient
                                # 5. Copy the generated key (32 characters, format: xxxx-xxxx-xxxx-xxxx)
                                # 6. Paste it here replacing YOUR_API_KEY_HERE
                                # 
                                # Troubleshooting:
                                # - If you get 401 errors, check if the key is active
                                # - Keys expire after 90 days by default
                                # - Ensure no extra spaces when pasting

BASE_URL = "https://api.example.com/v2"  # Default endpoint for US region
                                         # For EU: use https://eu.api.example.com/v2
                                         # For APAC: use https://apac.api.example.com/v2

# ========================================
# DATA EXTRACTION PARAMETERS
# ========================================
START_DATE = "2024-01-01"  # Modify as needed (format: YYYY-MM-DD)
END_DATE = "2024-12-31"    # Modify as needed (format: YYYY-MM-DD)
PAGE_SIZE = 100            # Number of records per API call (max: 100)

# [Rest of the implementation with full error handling, data validation, and CSV export]

## Important Notes
- **Never skip the web search phase** - it's crucial for understanding the connector
- **Always validate before saving** - ensure the script will work when credentials are added
- **Make scripts self-documenting** - users should understand every configuration option
- **Test logic flow** - even with placeholders, the script structure should be sound
- **Provide value** - the script should be ready to run once credentials are added

Remember: The goal is to create a script that a user can immediately understand and configure without needing to research the API documentation themselves. Your comments and placeholders should serve as a complete guide.
"""
react = create_react_agent(llm, tools=tools, prompt=system_prompt)


# Show title and description.
st.title("💬 Custom connector builder")
st.write("Welcome to custom connector builder. ")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Let me know your data requirement!"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    context = ""
    msgs = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))

    msgs.append(HumanMessage(content=prompt))
    output = None
    content = None
    for output in react.stream(
        {"messages": msgs},
        config={"thread_id": uuid.uuid1().hex, "recursion_limit": 100},
        stream_mode="values",
    ):
        last_message: AIMessage = output["messages"][-1]
        print(output["messages"])
        if isinstance(last_message, ToolMessage):
            content = last_message.content
            if last_message.name == "web_search":
                st.chat_message("assistant").write(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )

        if isinstance(last_message, AIMessage):
            try:

                # some hacks to seperate out tool call messages from AIMessage
                if isinstance(last_message.content, list):
                    if "text" in last_message.content[0]:
                        content = last_message.content[0]["text"]
                        st.chat_message("assistant").write(content)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": content}
                        )

                else:
                    content = last_message.content
                    st.chat_message("assistant").write(content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )
                if hasattr(last_message, "tool_calls"):
                    tool_calls = last_message.tool_calls
                    for tool_call in tool_calls:
                        #     if tool_call['name'] == "execute_shell_command_in_env":
                        #         msg=f"Running {tool_call['args']['command']}"
                        #         st.session_state.messages.append(
                        #     {"role": "assistant", "content": content}
                        # )
                        if tool_call["name"] == "save_code_to_file_in_env":
                            code = tool_call["args"]["code"]
                            st.code(code, language="python")

            except Exception as e:
                raise e
                breakpoint()

    st.session_state.messages.append({"role": "user", "content": prompt})
    if content:
        st.session_state.messages.append({"role": "assistant", "content": content})
