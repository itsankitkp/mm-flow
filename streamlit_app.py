import typing

try:
    # If NotRequired is missing (Python < 3.11), patch it from typing_extensions
    from typing_extensions import NotRequired

    typing.NotRequired = NotRequired
except ImportError:
    pass  # typing_extensions not installed → pip install typing_extensions


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

llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=st.secrets["ANTHROPIC_API_KEY"], max_tokens=20000)  # type: ignore
# llm = ChatOpenAI(model="gpt-5", api_key=st.secrets["OPENAI_API_KEY"])


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
model = ChatOpenAI(model="gpt-5-mini", api_key=st.secrets["OPENAI_API_KEY"])

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=368,
    max_summary_tokens=140,
    output_messages_key="llm_input_messages",
)


class State(AgentState):
    context: dict[str, RunningSummary]


today_date = datetime.datetime.now().isoformat()

system_prompt = """You are a data integration specialist that extracts data from ANY source and saves it as CSV.

## WORKFLOW

### 1. Identify Source Type
APIs (REST/GraphQL/SOAP) → Databases → Files (S3/FTP) → Web scraping

### 2. Research Documentation (MANDATORY)
Search: "[service] API documentation 2025", "[service] authentication"
Find: endpoints, auth methods, pagination, rate limits, required parameters

**Auth Priority: No auth > API Key > Basic Auth > OAuth2**
Get all credentials like access key, client ID/secret, before hand

### 3. Get Authentication Details

**Simple Auth:** Provide step-by-step from docs (where to find key, expected format)

**OAuth2:** 
- App creation steps from developer console
- Redirect URI: http://localhost:8080/callback (MANDATORY)
- Request client_id, client_secret, required scopes

### 4. Generate Code

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

**OAuth2 Structure:**
```python
#!pip install requests pandas
from serv import OAuthCallbackServer
from uuid import uuid4

# HARDCODE credentials
CLIENT_ID = "actual_id"
CLIENT_SECRET = "actual_secret"

class Connector:
    def __init__(self, client_id, client_secret, redirect_uri):
        '''Store credentials, initialize tokens to None'''
    
    def get_auth_url(self, state, scopes=None):
        '''Build authorization URL from docs, include offline_access'''
    
    def consent_handler(self, params):
        '''Extract code, exchange for tokens, store internally, return result'''
    
    def refresh_access_token(self):
        '''Use refresh_token to get new access_token'''
    
    def fetch_data(self):
        '''Fetch data using access_token, refresh if 401'''

# Execute OAuth flow
server = OAuthCallbackServer(host="localhost", port=8080)
connector = Connector(CLIENT_ID, CLIENT_SECRET, server.redirect_uri)
state = uuid4()
auth_url = connector.get_auth_url(state)
result = server.grant_consent(connector.consent_handler, timeout=120, expected_state=state)

if 'access_token' in result:
    data = connector.fetch_data()
    save_to_csv(data)
```
IN CASE OF OAUTH2, ONCE OAUTH FLOW CODE IS CREATED, RUN IT
AND SHOW AUTH URL TO USER SO THAT THEY CAN AUTHORIZE

### 5. Common Patterns
- **Pagination**: Implement based on docs (page/offset/cursor)
- **Rate limits**: Add delays/backoff as specified
- **Nested JSON**: Flatten before CSV
- **Token refresh**: Auto-refresh on 401

### 6. Validate
- Test with 1 row first
- Verify data structure matches user needs
- Ensure CSV output works
- Debug until successful

## RULES
- ALWAYS search latest 2025 docs
- ALWAYS test with 1 row first
- HARDCODE all credentials (no placeholders)
- OAuth2 redirect: http://localhost:8080/callback
- Choose simplest auth method available
- Include refresh token logic for OAuth2

## Success = 
✓ Docs researched
✓ Auth working (1 row test passes)
✓ Full data fetched
✓ CSV saved
"""


react = create_react_agent(
    llm,
    tools=tools,
    #                        pre_model_hook=summarization_node,
    # state_schema=State,
    prompt=system_prompt,
)


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
