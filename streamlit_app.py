import traceback
import typing


try:
    from typing_extensions import NotRequired

    typing.NotRequired = NotRequired
except ImportError:
    pass

import datetime
import re
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import streamlit as st
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
import uuid
from py_exc import tools
from upload import upload_file_to_mammoth
from langgraph.prebuilt.chat_agent_executor import AgentState
from todo import TODO_TOOLS, list_todos


# LLM Configuration
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=st.secrets["ANTHROPIC_API_KEY"],
    max_tokens=20000,
)

# Context Management Configuration - these can be made configurable via UI
MAX_CONTEXT_TOKENS = 20000
COMPACTION_THRESHOLD = 0.8


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
    """Perform a web search and return structured research results."""
    result = research_llm.invoke(query)
    return result


tools.append(web_search)
tools.extend(TODO_TOOLS)

today_date = datetime.datetime.now().isoformat()

SYSTEM_PROMPT = """You are a data integration specialist that extracts data from ANY source and saves it as CSV.

## WORKFLOW

### 1. Identify Source Type & Create Project Plan
APIs (REST/GraphQL/SOAP) → Databases → Files (S3/FTP) → Web scraping
**IMMEDIATELY create todos for the project:**
- Research documentation 
- Set up authentication
- Test connection with 1 row, fix if any error
- Implement full data extraction which stores all data to csv
- Save data as CSV
- Validate final output
SHOW ALL TODOS: call list_todos() after creating or completing any todo

### 2. Research Documentation (MANDATORY)
Search: "[service] API documentation 2025", "[service] authentication"
Find: endpoints, auth methods, pagination, rate limits, required parameters
**Auth Priority: No auth > API Key > Basic Auth > OAuth2**
Get all credentials like access key, client ID/secret, before hand
✅ **Check off research todo when complete**

### 3. Get Authentication Details & Update Progress
**Simple Auth:** Provide step-by-step from docs (where to find key, expected format)
**OAuth2:** 
- App creation steps from developer console
- Redirect URI: http://localhost:8080/callback (MANDATORY)
- Request client_id, client_secret, required scopes
✅ **Check off authentication setup todo when complete**

### 4. Generate Code & Track Progress
**Simple Auth Structure:**
```python
#!pip install requests pandas
# HARDCODE credentials
API_KEY = "actual_key_here"
def fetch_data():
    '''Test with 1 row, then fetch all data with pagination if needed'''
    
def save_to_csv(data):
    '''Convert to DataFrame and save as output.csv'''
```

### 5. Common Patterns & Update Status
- **Pagination**: Implement based on docs (page/offset/cursor)
- **Rate limits**: Add delays/backoff as specified
- **Nested JSON**: Flatten before CSV
- **Token refresh**: Auto-refresh on 401
✅ **Check off full data extraction todo when complete**

### 6. Validate & Final Cleanup
- Test with 1 row first
- Verify data structure matches user needs
- Ensure CSV output works
- Debug until successful
✅ **Check off CSV save and validation todos when complete**

## TODO MANAGEMENT RULES
- **ALWAYS** create todos at project start using create_todo
- **REGULARLY** use list_todos to show current progress
- **IMMEDIATELY** check_todo when steps are completed
- **DELETE** completed todos only when user requests
- Keep user informed of progress through todo updates
- Run code interactively. Save access tokens in secrets if needed so that re-auth is not needed on every run.

## RULES
- ALWAYS search latest 2025 docs
- ALWAYS test with 1 row first
- HARDCODE all credentials (no placeholders)
- OAuth2 redirect: http://localhost:8080/callback
- Choose simplest auth method available
- Include refresh token logic for OAuth2
- Use TODOs to track and show progress
- Show TODO list after every major step. Must at beginning, after research, after auth, after test, after full extraction, at end

## Success = 
✓ All todos completed and checked off
✓ Docs researched
✓ Auth working (1 row test passes)
✓ Full data fetched
✓ CSV saved and its absolute location is shared
"""

# Create enhanced React agent with context management
react = create_react_agent(
    llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


st.title("💬 Custom connector builder")
st.write("Welcome to custom connector builder with intelligent context management.")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Let me know your data requirement!"}
    ]

# Initialize thread ID for conversation continuity
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = uuid.uuid1().hex


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
        input={
            "messages": msgs,
        },
        config={"thread_id": st.session_state["thread_id"], "recursion_limit": 100},
        stream_mode="values",
    ):

        last_message = output["messages"][-1]
        print(last_message)
        if isinstance(last_message, ToolMessage):
            content = last_message.content
            if last_message.name == "web_search":
                st.chat_message("assistant").write(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )
            if "list_todos" in str(last_message.name):
                content = list_todos()
                content = content.replace("\n", "\n\n")
                st.chat_message("assistant").markdown(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )

        if isinstance(last_message, AIMessage):
            try:
                if isinstance(last_message.content, list):
                    if len(last_message.content) == 0:
                        continue
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
                        if tool_call["name"] == "save_code_to_file_in_env":
                            code = tool_call["args"]["code"]
                            st.code(code, language="python")

            except Exception as e:

                st.error(f"Error processing message: {traceback.format_exc()}")
                breakpoint()

    st.session_state.messages.append({"role": "user", "content": prompt})
