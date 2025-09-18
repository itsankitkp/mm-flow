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


from tools import SYSTEM_PROMPT, TOOLS
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

def show_csv(file_path: str) -> str:
    """Read and display a CSV data to user
    args:
        file_path (str): Absolute File Path to the CSV file
    """
    try:
        import pandas as pd

        df = pd.read_csv(file_path)
        st.dataframe(df)
        return f"Displayed CSV file: {file_path} with {len(df)} rows."
    except Exception as e:
        return f"Error displaying CSV file {file_path}: {e}"
    
tools.append(show_csv)

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