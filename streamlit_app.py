import typing
import typing_extensions

typing.NotRequired = typing_extensions.NotRequired


import traceback

import datetime
import re
import uuid
import os
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
    temperature=0,
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


def should_skip_content(content: str) -> bool:
    """Check if content should be skipped (raw tool calls, etc.)"""
    if not isinstance(content, str):
        return False

    content_lower = content.lower()
    return any(
        [
            "tool_use" in content_lower,
            content.strip().startswith("[{'id':"),
            content.strip().startswith("{'id':"),
            content.strip().startswith('[{"id":'),
            content.strip().startswith('{"id":'),
            len(content) > 500 and content.count("{") > 3 and content.count("}") > 3,
            "'type': 'tool_use'" in content,
            '"type": "tool_use"' in content,
        ]
    )


# Track CSV files created by the agent
def track_csv_file(file_path: str):
    """Track CSV files created by the agent for sidebar display"""
    if "csv_files" not in st.session_state:
        st.session_state["csv_files"] = []

    if file_path not in [f["path"] for f in st.session_state["csv_files"]]:
        file_name = os.path.basename(file_path)
        st.session_state["csv_files"].append(
            {
                "path": file_path,
                "name": file_name,
                "created_at": datetime.datetime.now().strftime("%H:%M:%S"),
            }
        )


def scan_for_csv_files(content: str):
    """Scan tool output content for CSV file paths and track them"""
    if not isinstance(content, str):
        return

    # Look for patterns like /tmp/.../*.csv or any path ending with .csv
    csv_path_pattern = r"(/tmp/[^/\s]+/[^/\s]*\.csv|/[^/\s]+/[^/\s]*\.csv|[^/\s]+\.csv)"
    matches = re.findall(csv_path_pattern, content)

    for match in matches:
        # Only track if the file actually exists
        if os.path.exists(match):
            track_csv_file(match)


def refresh_todos():
    """Fetch current todos and update session state"""
    try:
        todos_result = list_todos()
        if "todos" not in st.session_state:
            st.session_state["todos"] = []

        # Parse todos from the result string more intelligently
        if isinstance(todos_result, str):
            todos = []
            lines = todos_result.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("No todos")
                    and not line.startswith("Found")
                ):
                    # Look for lines that contain actual todo content
                    if any(
                        keyword in line.lower() for keyword in ["todo", "task", "id:"]
                    ):
                        # Clean up the line - remove extra formatting
                        clean_line = line.replace("- ", "").replace("* ", "").strip()
                        if clean_line:
                            todos.append(clean_line)

            st.session_state["todos"] = todos
    except Exception as e:
        st.session_state["todos"] = []


def check_and_refresh_todos(content: str):
    """Check if content mentions todos and refresh if needed"""
    if not isinstance(content, str):
        return

    content_lower = content.lower()
    # More aggressive detection for todo-related content
    todo_keywords = [
        "todo",
        "task",
        "created todo",
        "marked",
        "completed",
        "added todo",
    ]

    if any(keyword in content_lower for keyword in todo_keywords):
        refresh_todos()


def show_csv(file_path: str) -> str:
    """Read and display a CSV data to user
    args:
        file_path (str): Absolute File Path to the CSV file
    """
    try:
        import pandas as pd

        df = pd.read_csv(file_path)

        # Track the CSV file for sidebar
        track_csv_file(file_path)

        # Store CSV data in session state for display
        if "csv_data" not in st.session_state:
            st.session_state["csv_data"] = []

        st.session_state["csv_data"].append(
            {
                "path": file_path,
                "dataframe": df,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            }
        )

        return f"CSV_DISPLAY:{file_path}:{len(df)} rows"
    except Exception as e:
        return f"Error displaying CSV file {file_path}: {e}"


tools.append(show_csv)

thinking_placeholder = None


def pre_model_hook(state):
    """Show thinking indicator before LLM call"""
    st.session_state["processing"] = True
    st.session_state["current_step"] = "Agent is reasoning..."
    st.session_state["activity_log"].append(
        f"Thinking: {datetime.datetime.now().strftime('%H:%M:%S')}"
    )
    return state


def post_model_hook(state):
    """Remove thinking indicator after LLM call"""
    st.session_state["processing"] = False  # Set back to False
    st.session_state["current_step"] = "Processing response..."
    return state


# Create React agent
react = create_react_agent(
    llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    pre_model_hook=pre_model_hook,
    post_model_hook=post_model_hook,
)


def format_research_result(content: str) -> tuple[bool, str]:
    """Format ResearchResult objects nicely"""
    if "topic=" in content and "key_points=" in content:
        try:
            # Extract topic
            topic_match = re.search(r"topic='([^']*)'", content)
            if not topic_match:
                topic_match = re.search(r'topic="([^"]*)"', content)

            # Extract key points
            key_points_match = re.search(r"key_points=\[(.*?)\]", content, re.DOTALL)

            if topic_match and key_points_match:
                topic = topic_match.group(1)
                key_points_str = key_points_match.group(1)

                # Parse key points
                key_points = []
                for point in re.findall(r"'([^']*)'", key_points_str):
                    key_points.append(point)

                # Format nicely with better styling
                formatted = f"""
### 🔍 Research Results: {topic}

"""
                for i, point in enumerate(key_points, 1):
                    formatted += f"**{i}.** {point}\n\n"

                formatted += "---\n"

                return True, formatted
        except Exception:
            pass

    return False, content


def display_message_content(content: str, avatar: str = "🤖"):
    """Display message content with proper formatting"""
    if should_skip_content(content):
        return

    # Handle CSV display
    if content.startswith("CSV_DISPLAY:"):
        parts = content.split(":")
        if len(parts) >= 3:
            file_path = parts[1]
            row_info = parts[2]

            with st.chat_message("assistant", avatar="📊"):
                st.markdown(f"**📊 CSV Data: {file_path.split('/')[-1]}**")
                st.caption(f"📈 {row_info}")

                # Find and display the matching CSV data
                csv_found = False
                if "csv_data" in st.session_state:
                    for csv_item in st.session_state["csv_data"]:
                        if csv_item["path"] == file_path:
                            st.dataframe(
                                csv_item["dataframe"], use_container_width=True
                            )

                            # Show basic info
                            df = csv_item["dataframe"]
                            col1, col2, col3 = st.columns(3)

                            csv_found = True
                            break

                # If CSV data not found in session state, try to read directly
                if not csv_found and os.path.exists(file_path):
                    try:
                        import pandas as pd

                        df = pd.read_csv(file_path)
                        st.dataframe(df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Could not display CSV: {e}")
                elif not csv_found:
                    st.warning("CSV data not found")
        return

    # Check if it's a ResearchResult
    is_research, formatted_content = format_research_result(content)

    with st.chat_message("assistant", avatar=avatar):
        if is_research:
            st.markdown(formatted_content)
        elif "**" in content or "•" in content or "#" in content:
            st.markdown(content)
        else:
            st.write(content)


# Custom CSS for better styling
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
.csv-file-card {
    background-color: canvas;
    padding: 0.5rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border-left: 3px solid #28a745;
}
    
    .todo-card {
        background-color: #fff3cd;
        padding: 0.3rem 0.5rem;
        border-radius: 6px;
        margin-bottom: 0.3rem;
        border-left: 3px solid #ffc107;
        font-size: 0.85rem;
        line-height: 1.2;
    }
    
    .csv-empty-state, .todo-empty-state {
        background-color: #2d3748;
        border: 1px dashed #4a5568;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #a0aec0;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App Configuration
st.set_page_config(
    page_title="Data Integration Specialist",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with dynamic status container
sidebar_status_container = st.sidebar.empty()

with st.sidebar:
    st.markdown("### 🔗 Data Integration")
    st.markdown("---")

    # Session controls at the top
    if st.button("🔄 New Session", use_container_width=True):
        # Keep only the CSV data and suggested prompt if they exist
        csv_data = st.session_state.get("csv_data", [])
        suggested_prompt = st.session_state.get("suggested_prompt")

        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Restore what should persist
        if suggested_prompt:
            st.session_state["suggested_prompt"] = suggested_prompt

        st.rerun()

    st.markdown("---")

    # Status and thinking indicator - using dynamic container
    with sidebar_status_container.container():
        st.markdown("**Agent Status**")
        if st.session_state.get("processing", False):
            st.info("🧠 Thinking...")
            if "current_step" in st.session_state and st.session_state["current_step"]:
                st.caption(f"⚡ {st.session_state['current_step']}")
        else:
            st.success("✅ Ready")

    st.markdown("---")

    # CSV Files Section
    st.markdown("**📊 Generated CSV Files**")

    # Add a refresh button to re-scan for CSV files
    if st.button("🔄 Scan for CSV files", use_container_width=True):
        # Re-scan all messages for CSV files
        for msg in st.session_state.get("messages", []):
            if isinstance(msg["content"], str):
                scan_for_csv_files(msg["content"])
        st.rerun()

    if "csv_files" in st.session_state and st.session_state["csv_files"]:
        for csv_file in st.session_state["csv_files"]:
            st.markdown(
                f"""
            <div class="csv-file-card">
                <strong>📄 {csv_file['name']}</strong><br>
                <small>Created: {csv_file['created_at']}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Add download button
            try:
                if os.path.exists(csv_file["path"]):
                    with open(csv_file["path"], "rb") as file:
                        st.download_button(
                            label=f"⬇️ Download {csv_file['name']}",
                            data=file.read(),
                            file_name=csv_file["name"],
                            mime="text/csv",
                            key=f"download_{csv_file['path']}",
                            use_container_width=True,
                        )
                else:
                    st.caption("⚠️ File no longer exists")
            except Exception as e:
                st.caption(f"Download not available: {str(e)}")

            st.markdown("---")
    else:
        st.markdown(
            """
        <div class="csv-empty-state">
            📄 No CSV files detected yet<br>
            <small>Files will appear here when generated</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("**Quick Examples:**")
    examples = [
        "Get data from shopify",
        "Connect to Youtube data",
        "Process csv data from google drive",
        "Get contacts from xero",
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["suggested_prompt"] = f"Help me {example.lower()}"

    st.markdown("---")

    # Activity log
    if st.session_state.get("processing", False):
        st.markdown("**Recent Activity**")
        activity_log = st.session_state.get("activity_log", [])
        for activity in activity_log[-3:]:  # Show last 3 activities
            st.caption(f"• {activity}")


# Main header
st.markdown(
    """
<div class="welcome-card">
    <h1>🔗 Data Integration Specialist</h1>
    <p>Transform any data source into actionable insights with intelligent automation</p>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "**Welcome to Data Integration Specialist!** 🚀\n\nI'm here to help you connect to any data source and extract exactly what you need. Here's how I can assist:\n\n🔌 **Connect to APIs** - REST, GraphQL, webhooks\n🗄️ **Database Integration** - SQL, NoSQL, cloud databases  \n📊 **File Processing** - CSV, JSON, Excel, XML\n🔐 **Authentication** - OAuth, API keys, tokens\n\nJust describe what you want to achieve, and I'll build the solution for you!",
        }
    ]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = uuid.uuid1().hex

if "processing" not in st.session_state:
    st.session_state["processing"] = False

if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []

if "current_step" not in st.session_state:
    st.session_state["current_step"] = ""

if "csv_data" not in st.session_state:
    st.session_state["csv_data"] = []

if "csv_files" not in st.session_state:
    st.session_state["csv_files"] = []

if "todos" not in st.session_state:
    st.session_state["todos"] = []

# Auto-scan for CSV files and refresh todos on page load (only once per session)
if "initial_scan_done" not in st.session_state:
    st.session_state["initial_scan_done"] = True
    # Scan all existing messages for CSV files
    for msg in st.session_state.get("messages", []):
        if isinstance(msg["content"], str):
            scan_for_csv_files(msg["content"])

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            if "**" in msg["content"] or "•" in msg["content"]:
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])

# Handle suggested prompts from sidebar
if "suggested_prompt" in st.session_state:
    prompt = st.session_state["suggested_prompt"]
    del st.session_state["suggested_prompt"]
    st.rerun()

# Chat input
if not st.session_state.get("processing", False):
    prompt = st.chat_input(
        "💬 Describe your data integration needs...", key="main_input"
    )
else:
    st.chat_input("💬 Processing... please wait", disabled=True, key="disabled_input")
    prompt = None

if prompt:
    # Mark as processing immediately to update sidebar
    st.session_state["processing"] = True
    st.session_state["current_step"] = "Analyzing request"
    st.session_state["activity_log"].append(
        f"Started: {datetime.datetime.now().strftime('%H:%M:%S')}"
    )

    # Force sidebar refresh by triggering a rerun
    sidebar_status_container.empty()
    with sidebar_status_container.container():
        st.markdown("**Agent Status**")
        st.info("🧠 Thinking...")
        st.caption(f"⚡ {st.session_state['current_step']}")

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Prepare message history
    msgs = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))

    # Track displayed messages to avoid duplicates during streaming
    displayed_message_ids = set()
    final_assistant_content = []

    try:
        # Stream the agent response
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
            message_id = getattr(last_message, "id", None) or hash(
                str(last_message.content) + str(type(last_message))
            )

            # Only display if we haven't seen this message before
            if message_id not in displayed_message_ids:
                displayed_message_ids.add(message_id)

                print(last_message)

                if isinstance(last_message, AIMessage):
                    st.session_state["current_step"] = "Generating response"

                    if isinstance(last_message.content, list):
                        content = (
                            last_message.content[0].get(
                                "text", str(last_message.content)
                            )
                            if last_message.content
                            else ""
                        )
                    else:
                        content = last_message.content

                    if content and not should_skip_content(content):
                        display_message_content(content)
                        final_assistant_content.append(content)
                        # Real-time CSV scanning
                        scan_for_csv_files(content)
                        # Check for todos and refresh if mentioned

                elif isinstance(last_message, ToolMessage):
                    # Update step based on tool being used
                    tool_name = last_message.name
                    if "todo" in tool_name.lower():
                        st.session_state["current_step"] = "Managing tasks"
                    elif "web_search" in tool_name.lower():
                        st.session_state["current_step"] = "Searching web"
                    elif "file" in tool_name.lower() or "csv" in tool_name.lower():
                        st.session_state["current_step"] = "Processing files"
                    else:
                        st.session_state["current_step"] = f"Using {tool_name}"

                    st.session_state["activity_log"].append(f"Tool: {tool_name}")

                    content = last_message.content
                    if content and not should_skip_content(content):
                        # Special handling for different tool types
                        if "todo" in last_message.name.lower():
                            if "Created todo" in content or "Marked" in content:
                                with st.chat_message("assistant", avatar="✅"):
                                    st.success(content)
                                final_assistant_content.append(content)
                            # Refresh todos after any todo tool usage
                            refresh_todos()
                        elif "web_search" in last_message.name.lower():
                            display_message_content(content, "🔍")
                            final_assistant_content.append(content)
                        elif (
                            "show_csv" in last_message.name.lower()
                            or content.startswith("CSV_DISPLAY:")
                        ):
                            display_message_content(content, "📊")
                            final_assistant_content.append(content)
                        elif any(
                            keyword in last_message.name.lower()
                            for keyword in ["file", "code"]
                        ):
                            display_message_content(content, "📁")
                            final_assistant_content.append(content)
                        else:
                            display_message_content(content)
                            final_assistant_content.append(content)

                        # Real-time CSV scanning
                        scan_for_csv_files(content)
                        # Check for todos in tool messages too
                        check_and_refresh_todos(content)

    except Exception as e:
        st.session_state["current_step"] = "Error occurred"
        st.session_state["activity_log"].append(f"Error: {str(e)[:50]}...")

        with st.chat_message("assistant", avatar="❌"):
            st.error(f"Something went wrong: {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
        final_assistant_content.append(f"Error: {str(e)}")

    finally:
        # Clear processing indicator and state
        st.session_state["processing"] = False
        st.session_state["current_step"] = ""
        st.session_state["activity_log"].append(
            f"Completed: {datetime.datetime.now().strftime('%H:%M:%S')}"
        )

        # Keep activity log manageable
        if len(st.session_state["activity_log"]) > 10:
            st.session_state["activity_log"] = st.session_state["activity_log"][-10:]

        # Store final assistant response in session state (only once after streaming)
        if final_assistant_content:
            combined_content = "\n\n".join(final_assistant_content)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_content}
            )
            # Final refresh to ensure todos are current
            refresh_todos()

        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>💡 Tip: Be specific about your data source and requirements for best results</div>",
    unsafe_allow_html=True,
)
