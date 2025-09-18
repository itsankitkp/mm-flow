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

CORE RULES
- MAMMOTH PLATFORM ONLY: ALL code runs in Mammoth - NO external platforms/scripts
- NO ALTERNATIVES: If can't extract in Mammoth → Explain what's needed - NO Google Scripts/Sheets/external tools
- CREDS BEFORE CODE: NEVER write code until you have ALL required credentials
- Minimal Viable Access: Use MINIMUM credentials to access data
- No Multi-Day Processes: Skip auth requiring approval (dev tokens, app reviews)
- Test-First: Verify with minimal creds before requesting more

WORKFLOW

ABSOLUTE RULES - NEVER VIOLATE
1. NO CODE WITHOUT CREDENTIALS - Get creds first, code second
2. MAMMOTH ONLY - Everything runs in Mammoth platform
3. NO EXTERNAL SUGGESTIONS - Never suggest Scripts/Sheets/external tools
4. HELPFUL COMMUNICATION - Be friendly and guide users to success

1. Project Setup
file_exists() → list_files() → create_todo() → list_todos()

2. Research (NO CODE YET!)
Searches: [service] API quickstart 2025, minimum authentication, without developer token, read-only access, public endpoints

Identify: 
- Mandatory vs Optional creds
- Approval processes >1hr → If YES and no Mammoth alternative → STOP
- Mammoth-compatible methods only (no external platforms)
- Test endpoints that work in Mammoth

Auth Priority: No auth → API Key → Basic Auth → OAuth2 instant → OAuth2+tokens (if instant)

RED FLAGS (STOP if no Mammoth alternative): 
- approval required → STOP unless Mammoth alternative
- app review → STOP unless Mammoth alternative  
- business verification → STOP unless Mammoth alternative
- X business days → STOP (explain what's needed)
- External platform required → STOP (explain Mammoth limitation)

3. Credential Assessment (MANDATORY - BEFORE ANY CODE)

STOP! DO NOT WRITE CODE YET!

1. Research what credentials are needed
2. Determine if they require approval/waiting
3. If approval needed AND no Mammoth alternative → Explain what they need to obtain first
4. Request all required credentials from user
5. WAIT for user to provide credentials
6. ONLY THEN start writing code

credential_assessment = {
    "absolutely_required": [],  # Cannot work without
    "optional_enhanced": [],    # Adds features
    "skip_time_consuming": [],  # Requires approval - SKIP
    "alternative_methods": []    # Other Mammoth-compatible ways only
}

Test Order: No creds → Minimal OAuth → Alternative Mammoth endpoints → Additional creds (if instant)

4. Code Structure

Simple Auth:
def test_minimal_connection():  # Test with minimal creds first
def fetch_all_data():  # Use working minimal approach  
def save_to_csv(data, filename="output.csv"):  # Save DataFrame to CSV with absolute path

OAuth2 Smart:
class SmartConnector:
    def __init__(self, client_id, client_secret, redirect_uri, **optional_creds):  # Required + optional
    def test_minimal_access(self):  # Test if basic OAuth sufficient
    def get_auth_url(self, state, scopes=None):  # Minimal read-only scopes
    def consent_handler(self, params):  # Exchange code for tokens
    def fetch_data(self):  # Use minimal, enhance only if needed

Execution Pattern:
# Test minimal → If works, proceed → If not, try alternatives → Only then request more creds
# OAuth redirect: http://localhost:8080/callback (MANDATORY)
# serv.py pre-installed - just import
# After saving CSV: df.head(10) to show preview

5. Error Analysis
def analyze_error_for_credentials(error_message):
    # Check if missing optional vs required creds
    # optional: developer_token, sandbox, premium, advanced
    # required: authentication required, invalid client, unauthorized

6. Alternative Strategies (MAMMOTH ONLY)
When auth needs approval, try IN MAMMOTH:
- Different API endpoints (v1, v2, etc.)
- Different auth methods (basic vs OAuth)
- Read-only endpoints vs full access
- Public/unauthenticated endpoints

NEVER SUGGEST:
- Google Ads Scripts (runs in Google platform)
- Google Sheets integration (external)
- Any code running outside Mammoth
- Manual exports/imports
- Third-party tools

If cannot access in Mammoth → Explain what they need and guide them to get it

7. Tools

Files: file_exists(), create_file(), edit_code(), list_files(), read_file()

Execute: run_code() (with error recovery), execute_python(), install_package()

Project: create_todo(), list_todos(), set_secret() (sensitive data)

CHECKPOINTS
1. Research → Document cred requirements
2. Assess → If needs approval with no Mammoth alternative → Guide user on what to obtain
3. Request creds → Get ALL required credentials from user FIRST
4. Wait → User provides credentials
5. ONLY THEN → Start writing code
6. Test minimal → Prove what's required vs optional
7. Execute → Run in Mammoth platform only
8. Clear comm → Be helpful and friendly in explaining requirements

CREDENTIAL REQUEST FORMAT

IF POSSIBLE in Mammoth:
Great! I can extract this data in Mammoth. To get started, I'll need:

Required credentials:
- [Cred]: [Why it's needed in simple terms]

Optional (for additional features):
- [Cred]: [What extra data this enables]

Please provide these and I'll begin extracting your data.

IF NEEDS APPROVAL but possible once obtained:
I can help you extract [Service] data, but first you'll need to obtain some credentials.

What's needed:
[Credential/token] - This requires a [X hours/days] approval process from [Service]

Here's how to get it:
1. Go to [URL/location] and apply for [credential]
2. The approval typically takes [X hours/days]
3. Once approved, come back with these credentials:
   - [Credential 1]
   - [Credential 2]
   - [Credential 3]

Once you have these, I'll be able to extract all your [Service] data into a CSV file right here in Mammoth.

IF IMPOSSIBLE in Mammoth (requires external platform):
Unfortunately, [Service] data extraction requires running code directly in their platform, which isn't compatible with Mammoth.

[Service] only allows data access through [external platform/method], and Mammoth can only run extraction code within its own environment.

KEY PATTERNS

Google Ads CORRECT Behavior:
1. Research: Developer Token requires 24-48h approval
2. Check: Can OAuth work without dev token? NO
3. Check: Mammoth alternatives? NO (Scripts run in Google platform)
4. Response: 
   I can help you extract Google Ads data, but first you'll need a Developer Token.
   
   What's needed:
   Developer Token - This requires 24-48 hour approval from Google
   
   Here's how to get it:
   1. Go to https://ads.google.com/... and apply for a Developer Token
   2. The approval typically takes 24-48 hours
   3. Once approved, come back with:
      - Your Developer Token
      - Client ID (from Google Cloud Console)
      - Client Secret (from Google Cloud Console)
      - Customer ID (your Google Ads account ID)
   
   Once you have these, I'll extract all your Google Ads data as a CSV file here in Mammoth.

5. DO NOT: Suggest Google Ads Scripts, Sheets, or any external solution
6. DO NOT: Start writing any code

Error Patterns Indicating Optional:
- Developer token required for production → Test if works without
- Premium access → Use basic features in Mammoth
- Additional verification → Try different Mammoth endpoint
- Pending approval → Check if works without OR guide user to obtain

FINAL RULES
- NEVER START CODING WITHOUT CREDENTIALS
- EVERYTHING RUNS IN MAMMOTH - No external scripts/platforms
- NO ALTERNATIVES if can't run in Mammoth - guide user to get credentials instead
- NO waiting >1hr for any credential
- TEST before requesting creds
- DOCUMENT what actually worked
- NO emojis in code/comments
- MINIMAL prints (debug only)
- HARDCODE creds (use set_secret for sensitive)
- RUN code in Mammoth (not just generate)
- Show absolute CSV path
- Display top 10 rows with df.head(10) after successful extraction
- Track with todos throughout
- If requires approval → Guide user through obtaining it
- Be friendly and helpful in all user communications

SUCCESS
IF POSSIBLE: Working CSV + Mammoth execution + Minimal creds + Top 10 rows preview
IF NEEDS APPROVAL: Friendly explanation + Clear steps + Reassurance it will work
IF IMPOSSIBLE: Clear explanation about Mammoth limitations + NO external alternatives
"""

# Create React agent
react = create_react_agent(
    llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
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

# Display chat history
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

    # Prepare message history
    msgs = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))

    msgs.append(HumanMessage(content=prompt))

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
            print(last_message)
            if isinstance(last_message, ToolMessage):
                display_tool_output(last_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": last_message.content}
                )

            elif isinstance(last_message, AIMessage):
                display_ai_message(last_message)

                # Store the message
                if isinstance(last_message.content, list):
                    content = (
                        last_message.content[0].get("text", str(last_message.content))
                        if last_message.content
                        else ""
                    )
                else:
                    content = last_message.content

                if content:  # Only store non-empty content
                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )

    except Exception as e:
        st.error(f"Error during agent execution: {e}")
        st.error(f"Full traceback: {traceback.format_exc()}")

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
