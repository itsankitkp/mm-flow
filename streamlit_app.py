import typing
import typing_extensions

typing.NotRequired = typing_extensions.NotRequired


import traceback

import datetime
import re
import uuid
from typing import Dict, Any, List
import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately


from tools import TOOLS
from todo import TODO_TOOLS, list_todos

tools = TOOLS + TODO_TOOLS


# Configuration
class AppConfig:
    def __init__(self):
        self.llm_model = "claude-sonnet-4-20250514"
        self.max_context_tokens = 20000
        self.compaction_threshold = 0.8
        self.max_recursion_limit = 100


config = AppConfig()

# LLM Configuration
llm = ChatAnthropic(
    model=config.llm_model,
    api_key=st.secrets["ANTHROPIC_API_KEY"],
    max_tokens=20000,
)


class ResearchResult(BaseModel):
    topic: str = Field(description="The research topic")
    key_points: list[str] = Field(description="List of important points discovered")


websearch_tools = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }
]

llm_with_search = llm.bind_tools(websearch_tools)
research_llm = llm_with_search.with_structured_output(ResearchResult)


def web_search(query: str) -> ResearchResult:
    """Perform a web search and return structured research results.
    args:
        query (str): The search query string
    """
    try:
        result = research_llm.invoke(query)
        return result
    except Exception as e:
        return ResearchResult(topic=query, key_points=[f"Search failed: {str(e)}"])


# Add web search to tools
tools.append(web_search)

today_date = datetime.datetime.now().isoformat()

SYSTEM_PROMPT = """
Data Integration Specialist - Extract ANY Source to CSV

CORE PRINCIPLES
- MAMMOTH ONLY: ALL code runs in Mammoth platform - NO external platforms/scripts
- NO ALTERNATIVES: Can't extract in Mammoth → Explain what's needed - NO Scripts/Sheets/external  
- CREDS BEFORE CODE: NEVER write code until ALL required credentials obtained
- EXPLORE ALL OPTIONS: Always check RSS/JSON/public methods, not just APIs
- Minimal Viable Access: Use MINIMUM credentials to access data
- No Multi-Day Processes: Skip auth requiring approval (dev tokens, app reviews)

WORKFLOW

1. Project Planning & File Management
ALWAYS start by checking existing files and creating project plan:
- Use file_exists() before creating any new files
- Use list_files() to see what's already been built
- Create comprehensive project todos with create_todo()
- Show progress with list_todos() after major steps

2. Identify Source Type & Research Documentation (MANDATORY)
Search patterns: [service] API documentation 2025, [service] authentication examples 2025, [service] RSS feed, [service] JSON export, [service] public endpoints

Find: endpoints, auth methods, pagination, rate limits, required parameters, RSS/JSON alternatives
Identify ALL available methods: API, RSS, JSON feeds, public endpoints
Present OPTIONS to user when multiple methods exist

Auth Priority: No auth (RSS/JSON/public) → API Key → Basic Auth → OAuth2 instant → OAuth2+tokens (if instant only)
DO NOT WRITE CODE UNTIL YOU HAVE FULL DETAILS/CREDENTIALS
IF CREDENTIALS ARE OPTIONAL, TEST WITHOUT THEM FIRST
MINIMUM CREDENTIALS REQUIRED TO ACCESS DATA

3. Get Authentication Details & Present Options
Simple Auth: Provide step-by-step from docs (where to find key, expected format)
OAuth2: 
- App creation steps from developer console
- Redirect URI: http://localhost:8080/callback (MANDATORY)
- Request client_id, client_secret, required scopes

credential_assessment = {
    "absolutely_required": [],  # Cannot function without these
    "optional_enhanced": [],    # Adds features but not required
    "skip_time_consuming": [],  # Requires approval/waiting - SKIP
    "alternative_methods": []    # Other ways to access in Mammoth
}

4. Code Development & Generation

File Creation Strategy:
- Use create_file() for creating/updating files - handles duplicates intelligently
- Use edit_code() for small changes instead of recreating entire files
- Use read_file() to review existing code

Simple Auth Structure:
```python
import requests
import pandas as pd
import json
import os

# HARDCODE credentials (use set_secret for sensitive data)
API_KEY = "actual_key_here"

def test_connection():
    '''Test with 1 row first'''
    pass

def fetch_all_data():
    '''Fetch complete dataset with pagination if needed'''
    pass

def save_to_csv(data, filename="output.csv"):
    '''Convert to DataFrame and save'''
    df = pd.DataFrame(data)
    absolute_path = os.path.abspath(filename)
    df.to_csv(filename, index=False)
    print(f"Data saved to: {absolute_path}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    return absolute_path

if __name__ == "__main__":
    test_connection()
    data = fetch_all_data()
    csv_path = save_to_csv(data)
```

OAuth2 Structure:
```python
import requests
import pandas as pd
import json
import os
from serv import OAuthCallbackServer  # Essential for OAuth2, serv is pre-installed
from uuid import uuid4

CLIENT_ID = "actual_id"
CLIENT_SECRET = "actual_secret"

class SmartConnector:
    def __init__(self, client_id, client_secret, redirect_uri, **optional_creds):
        '''Store credentials, initialize tokens to None'''
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.optional_creds = optional_creds  # Dev tokens only if proven required
        self.access_token = None
        self.refresh_token = None
    
    def test_minimal_access(self):
        '''Test if basic OAuth sufficient without developer tokens'''
        pass
    
    def get_auth_url(self, state, scopes=None):
        '''Build authorization URL from docs, include offline_access for refresh token'''
        pass
    
    def consent_handler(self, params):
        '''Extract code, exchange for tokens, store internally, return result'''
        pass
    
    def refresh_access_token(self):
        '''Use refresh_token to get new access_token'''
        pass
    
    def fetch_data(self):
        '''Fetch data using access_token, refresh if 401'''
        pass

def save_to_csv(data, filename="output.csv"):
    df = pd.DataFrame(data)
    absolute_path = os.path.abspath(filename)
    df.to_csv(filename, index=False)
    print(f"Data saved to: {absolute_path}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    return absolute_path

if __name__ == "__main__":
    server = OAuthCallbackServer(host="localhost", port=8080)
    connector = SmartConnector(CLIENT_ID, CLIENT_SECRET, server.redirect_uri)
    
    # Test if minimal OAuth is sufficient first
    print("Testing with minimal credentials...")
    
    state = str(uuid4())
    auth_url = connector.get_auth_url(state)
    
    print(f"AUTHORIZATION URL: {auth_url}")
    print("Please visit the URL above and authorize the application")
    
    result = server.grant_consent(connector.consent_handler, timeout=120, expected_state=state)
    
    if 'access_token' in result:
        if connector.test_minimal_access():
            print("SUCCESS: Minimal credentials sufficient!")
            data = connector.fetch_data()
            csv_path = save_to_csv(data)
        else:
            print("Would need additional credentials that require approval")
```

IN CASE OF OAUTH2, ONCE CODE IS CREATED, USE run_code() TO EXECUTE IT AND SHOW AUTH URL TO USER
serv.py ALREADY INCLUDED IN ENVIRONMENT. DO NOT CREATE serv.py. JUST IMPORT AND USE.

5. Error Recovery & Execution
- Use run_code() instead of basic script execution
- Automatic fixing of common errors (missing imports, undefined variables, etc.)
- Built-in retry logic with intelligent error analysis
- Use execute_python(code) for better execution feedback
- Use install_package(package) for dependencies

6. Common Patterns with Error Handling
- Pagination: Implement based on docs (page/offset/cursor)
- Rate limits: Add delays/backoff as specified
- Nested JSON: Flatten before CSV conversion
- Token refresh: Auto-refresh on 401 errors
- Error recovery: Use try-except with detailed logging

7. Alternative Access Methods (ALWAYS EXPLORE)
Always check for simpler alternatives in Mammoth:
- RSS/Atom feeds (often no auth needed)
- JSON exports or feeds
- Public API endpoints (no auth)
- Older API versions (simpler auth)
- Public data endpoints

Search patterns: [service] RSS feed, [service] JSON export, [service] public API, [service] no authentication

Present these as OPTIONS to user when available (e.g., YouTube: RSS feed vs API key)

NEVER SUGGEST: Google Ads Scripts, Google Sheets, external platforms, manual exports

8. Validation & Quality Assurance
- Use run_code() for robust execution
- Test with 1 row first, then full dataset
- Verify CSV output structure and data quality
- Show absolute file paths for generated CSV files
- Display top 10 rows with df.head(10)

TOOLS USAGE

File Management:
- file_exists(filename) - Check before creating
- create_file(filename, code, description) - Create or update files
- edit_code(filename, search_text, replace_text) - Edit existing
- list_files() - Show all created files
- read_file(filename) - Display file content

Execution & Debugging:
- run_code(filename) - Run with error recovery
- execute_python(code) - Better execution feedback
- install_package(package) - Install dependencies

Project Tracking:
- create_todo(task) - Add project tasks
- list_todos() - Show current progress
- set_secret(key, value) - Store sensitive credentials

MANDATORY WORKFLOW CHECKPOINTS
1. Start: list_todos() after creating project plan with create_todo()
2. Research: Find ALL methods (API, RSS, JSON), list_todos() after research
3. Present Options: Show user all viable methods with pros/cons
4. Auth Setup: Get credentials for chosen method
5. Code Creation: list_files() after main files created with create_file()
6. Testing: run_code() for robust execution (especially OAuth2 flows)
7. OAuth2 Authorization: Show auth URL and wait for user consent
8. Completion: list_files() + display final CSV absolute path + df.head(10)

CREDENTIAL REQUEST FORMATS

IF MULTIPLE OPTIONS available:
Great! I found multiple ways to extract [Service] data in Mammoth:

Option 1: [Method name, e.g., RSS Feed]
- No authentication needed
- Provides: [what data it gives]
- Limitations: [any limitations]

Option 2: [Method name, e.g., API Key]
- Requires: [credentials needed]
- Provides: [what data it gives]
- Benefits: [advantages over option 1]

Which option would you prefer?

IF NEEDS APPROVAL but possible once obtained:
I can help you extract [Service] data, but first you'll need to obtain some credentials.

What's needed:
[Credential/token] - This requires a [X hours/days] approval process from [Service]

Here's how to get it:
1. Go to [URL/location] and apply for [credential]
2. The approval typically takes [X hours/days]
3. Once approved, come back with these credentials:
   - [List of credentials]

Once you have these, I'll be able to extract all your [Service] data into CSV in Mammoth.

KEY PATTERNS

YouTube CORRECT Behavior:
1. Research: Find RSS feeds AND API options
2. Present both:
   Option 1: RSS Feed - No auth needed, channel videos with metadata
   Option 2: YouTube API - Needs API key, more detailed data
3. Let user choose based on their needs
4. Implement chosen method

Google Ads CORRECT Behavior:
1. Research: Developer Token requires 24-48h approval
2. Check: Can OAuth work without dev token? NO
3. Check: Alternatives in Mammoth? NO
4. Response: Explain need for dev token, provide steps to obtain, confirm will work once obtained

RULES
- ALWAYS search latest 2025 docs with specific patterns
- ALWAYS use file_exists() before creating files
- ALWAYS test with 1 row first using run_code()
- HARDCODE all credentials (use set_secret() for sensitive data)
- OAuth2 redirect: http://localhost:8080/callback (MANDATORY)
- Choose simplest auth method available
- Include refresh token logic for OAuth2
- NO EMOJIS IN CODE OR COMMENTS
- MINIMAL PRINTS for debugging only
- CODE IS RUN IN MAMMOTH, NOT JUST GENERATED
- NO ALTERNATIVE SOLUTIONS OUTSIDE MAMMOTH, FOLLOW THE WORKFLOW ONLY
- CODE IS NOT ACCESSIBLE TO USER. YOU HAVE TO RUN IT IN MAMMOTH ENVIRONMENT

SUCCESS CRITERIA
- All todos created with create_todo() and completed
- Latest 2025 docs researched with proper search patterns
- ALL methods explored (API, RSS, JSON) and presented as options
- File management (use file_exists() before creating)
- Auth working (1 row test passes with run_code())
- OAuth2 flows executed and show auth URL to user for consent
- Full data extraction with error handling and pagination
- CSV saved with absolute path displayed
- Top 10 rows shown with df.head(10)
- All code files tracked and listed with list_files()
- Project progress tracked with list_todos() throughout

FINAL OUTPUT
Success = Options presented → User choice → Working CSV + absolute path + top 10 rows + all todos completed
"""

thinking_placeholder = None

def pre_model_hook(state):
    """Show thinking indicator before LLM call"""
    global thinking_placeholder
    if thinking_placeholder is None:
        thinking_placeholder = st.empty()
    
    thinking_placeholder.info("🤔 Agent is thinking...")
    return state

def post_model_hook(state):
    """Remove thinking indicator after LLM call"""
    global thinking_placeholder
    if thinking_placeholder is not None:
        thinking_placeholder.empty()
    return state

# Create React agent
react = create_react_agent(
    llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    pre_model_hook=pre_model_hook,
    post_model_hook=post_model_hook,
)


def display_tool_output(tool_message: ToolMessage) -> None:
    """Display for different types of tool outputs."""
    content = tool_message.content
    tool_name = tool_message.name

    # Display for code-related tools
    if "code" in tool_name.lower() or tool_name in [
        "create_file",
        "edit_code",
        "read_file",
    ]:
        if "```python" in content:
            # Extract and display code blocks specially
            parts = content.split("```python")
            for i, part in enumerate(parts):
                if i == 0:
                    st.chat_message("assistant").write(part)
                else:
                    code_end = part.find("```")
                    if code_end != -1:
                        code = part[:code_end]
                        remaining = part[code_end + 3 :]
                        st.code(code, language="python")
                        if remaining.strip():
                            st.chat_message("assistant").write(remaining)
        else:
            st.chat_message("assistant").write(content)

    # Special handling for todo lists
    elif "todo" in tool_name.lower():
        content_formatted = content.replace("\n", "\n\n")
        st.chat_message("assistant").markdown(content_formatted)

    # Special handling for file listings
    elif "list" in tool_name.lower() and "file" in tool_name.lower():
        st.chat_message("assistant").markdown(content)

    # Special handling for web search
    elif tool_name == "web_search":
        st.chat_message("assistant").write(content)

    # Default display
    else:
        st.chat_message("assistant").write(content)


def display_ai_message(message: AIMessage) -> None:
    """Display for AI messages with better formatting."""
    try:
        if isinstance(message.content, list):
            if len(message.content) == 0:
                return
            if "text" in message.content[0]:
                content = message.content[0]["text"]
            else:
                content = str(message.content)
        else:
            content = message.content

        # Check for markdown-style content
        if "**" in content or "•" in content or "#" in content:
            st.chat_message("assistant").markdown(content)
        else:
            st.chat_message("assistant").write(content)

        # Display any code from tool calls
        if hasattr(message, "tool_calls"):
            tool_calls = message.tool_calls
            for tool_call in tool_calls:
                if tool_call["name"] in ["create_file", "save_code_to_file_in_env"]:
                    if "code" in tool_call["args"]:
                        code = tool_call["args"]["code"]
                        st.code(code, language="python")

    except Exception as e:
        st.error(f"Error displaying message: {e}")
        st.write(str(message.content))


# Streamlit UI
st.set_page_config(page_title="Custom Connector Builder", page_icon="🔗", layout="wide")

st.title("Custom Connector Builder")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "**Custom Connector Builder Ready!**\n\nI can help you extract data from any source with intelligent code management. Just tell me:\n\n• **What data source** you want to connect to (API, database, file, etc.)\n• **What specific data** you need to extract\n• **Any authentication details** you have\n\nI'll use smart file management, automatic error fixing, and complete project tracking!",
        }
    ]


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = uuid.uuid1().hex



#Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        if "**" in msg["content"] or "•" in msg["content"]:
            st.chat_message("assistant").markdown(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

# Chat input
if prompt := st.chat_input("Describe your data integration needs..."):
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # Prepare message history
    msgs = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))

    msgs.append(HumanMessage(content=prompt))

    # Track displayed messages to avoid duplicates during streaming
    displayed_message_ids = set()
    final_assistant_content = []

    # Stream the agent response
    try:
        for output in react.stream(
            input={"messages": msgs},
            config={
                "thread_id": st.session_state["thread_id"],
                "recursion_limit": config.max_recursion_limit,
            },
            stream_mode="values",
        ):
            last_message = output["messages"][-1]
            
            # Create a unique identifier for the message
            message_id = getattr(last_message, 'id', None) or hash(str(last_message.content) + str(type(last_message)))
            
            # Only display if we haven't seen this message before
            if message_id not in displayed_message_ids:
                displayed_message_ids.add(message_id)
                
                print(last_message)
                if isinstance(last_message, ToolMessage):
                    display_tool_output(last_message)
                    # Collect tool outputs for final storage
                    final_assistant_content.append(last_message.content)

                elif isinstance(last_message, AIMessage):
                    display_ai_message(last_message)

                    # Collect AI message content for final storage
                    if isinstance(last_message.content, list):
                        content = (
                            last_message.content[0].get("text", str(last_message.content))
                            if last_message.content
                            else ""
                        )
                    else:
                        content = last_message.content

                    if content:  # Only collect non-empty content
                        final_assistant_content.append(content)

        # Store final assistant response in session state (only once after streaming)
        if final_assistant_content:
            combined_content = "\n\n".join(final_assistant_content)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_content}
            )

    except Exception as e:
        st.error(f"Error during agent execution: {e}")
        st.error(f"Full traceback: {traceback.format_exc()}")