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

def show_csv(file_path: str) -> str:
    """Read and display a CSV data to user
    args:
        file_path (str): Absolute File Path to the CSV file
    """
    try:
        import pandas as pd

        df = pd.read_csv(file_path)
        
        # Store CSV data in session state for display
        if "csv_data" not in st.session_state:
            st.session_state["csv_data"] = []
        
        st.session_state["csv_data"].append({
            "path": file_path,
            "dataframe": df,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        })
        
        return f"CSV_DISPLAY:{file_path}:{len(df)} rows"
    except Exception as e:
        return f"Error displaying CSV file {file_path}: {e}"
    
tools.append(show_csv)
thinking_placeholder = None

def pre_model_hook(state):
    """Show thinking indicator before LLM call"""
    st.session_state['processing'] = True
    return state

def post_model_hook(state):
    """Remove thinking indicator after LLM call"""
    st.session_state['processing'] = False
    return state
# Create React agent 
react = create_react_agent(
    llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
        pre_model_hook=pre_model_hook,
    post_model_hook=post_model_hook,
)


def should_skip_content(content: str) -> bool:
    """Check if content should be skipped (raw tool calls, etc.)"""
    if not isinstance(content, str):
        return False
    
    content_lower = content.lower()
    return any([
        "tool_use" in content_lower,
        content.strip().startswith("[{'id':"),
        content.strip().startswith("{'id':"),
        content.strip().startswith("[{\"id\":"),
        content.strip().startswith("{\"id\":"),
        len(content) > 500 and content.count("{") > 3 and content.count("}") > 3
    ])


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
                
                # Format nicely
                formatted = f"## 🔍 {topic}\n\n"
                for i, point in enumerate(key_points, 1):
                    formatted += f"{i}. {point}\n\n"
                
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
                
                # Display the most recent CSV data
                if "csv_data" in st.session_state and st.session_state["csv_data"]:
                    latest_csv = st.session_state["csv_data"][-1]
                    st.dataframe(latest_csv["dataframe"], use_container_width=True)
                    
                    # Show basic info
                    df = latest_csv["dataframe"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
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
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# App Configuration
st.set_page_config(
    page_title="Data Integration Specialist", 
    page_icon="🔗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.markdown("### 🔗 Data Integration")
    st.markdown("---")
    
    # Status and thinking indicator
    status_container = st.container()
    with status_container:
        st.markdown("**Agent Status**")
        if st.session_state.get("processing", False):
            st.info("🧠 Thinking...")
            if "current_step" in st.session_state:
                st.caption(f"⚡ {st.session_state['current_step']}")
        else:
            st.success("✅ Ready")
    
    st.markdown("---")
    
    st.markdown("**Quick Examples:**")
    examples = [
        "Get data from shopify",
        "Connect to Youtube data", 
        "Process csv data from google drive",
        "Get contacts from xero"
    ]
    
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["suggested_prompt"] = f"Help me {example.lower()}"
    
    st.markdown("---")
    
    # Session controls
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
    
    # Activity log
    if st.session_state.get("processing", False):
        st.markdown("**Recent Activity**")
        activity_log = st.session_state.get("activity_log", [])
        for activity in activity_log[-3:]:  # Show last 3 activities
            st.caption(f"• {activity}")


# Main header
st.markdown("""
<div class="welcome-card">
    <h1>🔗 Data Integration Specialist</h1>
    <p>Transform any data source into actionable insights with intelligent automation</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "**Welcome to Mammoth Integration Specialist!** 🚀\n\nI'm here to help you connect to any data source and extract exactly what you need. Here's how I can assist:\n\n🔌 **Connect to APIs** - REST, GraphQL, webhooks\n🗄️ **Database Integration** - SQL, NoSQL, cloud databases  \n📊 **File Processing** - CSV, JSON, Excel, XML\n🔐 **Authentication** - OAuth, API keys, tokens\n\nJust describe what you want to achieve, and I'll build the solution for you!",
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
    prompt = st.chat_input("💬 Describe your data integration needs...", key="main_input")
else:
    st.chat_input("💬 Processing... please wait", disabled=True, key="disabled_input")
    prompt = None

if prompt:
    # Mark as processing
    st.session_state["processing"] = True
    st.session_state["current_step"] = "Analyzing request"
    st.session_state["activity_log"].append(f"Started: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
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

    # Collect all assistant responses
    all_responses = []
    step_count = 0
    
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
            print(last_message)
            step_count += 1
            
            if isinstance(last_message, AIMessage):
                st.session_state["current_step"] = "Generating response"
                
                if isinstance(last_message.content, list):
                    content = (
                        last_message.content[0].get("text", str(last_message.content))
                        if last_message.content
                        else ""
                    )
                else:
                    content = last_message.content
                
                if content and not should_skip_content(content):
                    display_message_content(content)
                    all_responses.append(content)
            
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
                            all_responses.append(content)
                    elif "web_search" in last_message.name.lower():
                        display_message_content(content, "🔍")
                        all_responses.append(content)
                    elif "show_csv" in last_message.name.lower() or content.startswith("CSV_DISPLAY:"):
                        display_message_content(content, "📊")
                        all_responses.append(content)
                    elif any(keyword in last_message.name.lower() for keyword in ["file", "code"]):
                        display_message_content(content, "📁")
                        all_responses.append(content)
                    else:
                        display_message_content(content)
                        all_responses.append(content)
    
    except Exception as e:
        st.session_state["current_step"] = "Error occurred"
        st.session_state["activity_log"].append(f"Error: {str(e)[:50]}...")
        
        with st.chat_message("assistant", avatar="❌"):
            st.error(f"Something went wrong: {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
        all_responses.append(f"Error: {str(e)}")
    
    finally:
        # Clear processing indicator and state
        st.session_state["processing"] = False
        st.session_state["current_step"] = ""
        st.session_state["activity_log"].append(f"Completed: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Keep activity log manageable
        if len(st.session_state["activity_log"]) > 10:
            st.session_state["activity_log"] = st.session_state["activity_log"][-10:]
        
        # Store combined response in session state
        if all_responses:
            combined_response = "\n\n".join(all_responses)
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": combined_response
            })
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>💡 Tip: Be specific about your data source and requirements for best results</div>", 
    unsafe_allow_html=True
)