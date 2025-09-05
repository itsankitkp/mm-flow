import datetime
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
        "max_uses": 10,
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
tools.append(upload_file_to_mammoth)

# system_prompt = """
# You are expert in data and python coding. Your job is to write custom script to pull data,
# generate csv and upload to Mammoth.
# Following following steps in order:
# 1. Understand the user request. 
# 2. Understand what data is needed to fulfill the request, filters like time range if needed involved.
# 3. Use web tool to find relevant data sources and research about data thoroughly. 
# 4. If it is not specified, ask user for more details while giving relevant suggestions from insights gained in step 3.
# eg how to setup keys, which data to choose based intent, which filtering may be needed. Be helpful.
# 5. Once you have all the information, write a script to pull the data, generate a CSV, and upload it to Mammoth.
# 6. Repeat step 5 until script is valid and works.
# 7. Make sure to use the upload_file_to_mammoth tool to upload the generated CSV
# 8. Finally, inform the user that the file has been uploaded successfully.

# Note: Common packages like requests, pandas, numpy are already installed
# ALWAYS start with pip install of dependencies if you need to install any packages.
# ALWAYS import packages before using them.
# You can only do things programmatically, you don't have access to shell (other than for pip install)
# """

today_date = datetime.datetime.now().isoformat()
system_prompt = f"""
You are an expert in data engineering and Python programming, specializing in data extraction, and integration with Mammoth platform.

## Core Objective
Write custom Python scripts to extract data from various sources, generate CSV files, and upload 
them to Mammoth based on user requirements.

## Process Workflow

### 1. Requirements Analysis
- Parse and understand the user's data request thoroughly
- Identify key parameters: data types, sources, time ranges, filters, credential requirements and output format
- Clarify any ambiguities before proceeding

### 2. Data Source Research
- Use web tools to research and identify appropriate data sources
- Evaluate API documentation, authentication requirements, and data availability
- Document rate limits, data freshness, and any access restrictions

### 3. User Consultation (if needed)
- Request missing specifications with concrete suggestions based on research
- Provide examples of:
  * API key setup procedures
  * Available data fields and filtering options
  * Recommended time ranges and batch sizes
  * Data format preferences
Limitations: Auth2.0 based authentications are out of scope for now.

### 4. Script Development
- Write modular, well-commented Python code following these principles:
  * Include comprehensive error handling with try-except blocks
  * Implement data validation and sanitization
  * Use efficient data processing techniques (chunking for large datasets)
  * Include retry logic for network requests

### 5. Testing and Validation
- Test the script with sample data first
- Validate CSV structure and data integrity
- Handle edge cases (empty results, malformed data, connection failures)
- Iterate until the script executes successfully

### 6. File Upload
- Generate the final CSV
- Use the upload_file_to_mammoth tool to upload the file
- Verify successful upload and provide confirmation

### 7. Completion Report
- Confirm successful upload

## Technical Guidelines

### Environment Setup
- Common packages available: requests, pandas, numpy, datetime, json, csv
- For additional packages: Start with `pip install <package>` command
- Always import required modules at the beginning of the script

### Best Practices
- **Data Security**: hardcode values like API keys
- **Performance**: Implement pagination for large datasets; use batch processing where appropriate
- **Error Resilience**: Include graceful error handling and informative error messages
- **Documentation**: DO NOT WRITE DOCSTRING OR COMMENTS
- **Data Quality**: Validate data types, handle missing values, and ensure CSV compatibility

### Constraints
- No direct shell access (except for pip installations)
- All operations must be performed programmatically
YOUR JOB IS TO FINAL PUSH to MAMMOTH USING upload_file_to_mammoth TOOL. This can not be 
done unless all other steps are complete and script is working.
ALWAYS USE CURRENT YEAR as per as {today_date}. ALWAYS REFER LATEST CONTENT WHILE GIVING STEPS TO USER.
"""
react = create_react_agent(llm, tools=tools, prompt=system_prompt)

# Show title and description.
st.title("💬 Custom connector builder")
st.write(
    "Welcome to custom connector builder. "
)


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
    content=None
    for output in react.stream(
        {"messages": msgs},
        config={"thread_id": uuid.uuid1().hex, "recursion_limit": 100},
        stream_mode="values",
    ):
        last_message: AIMessage = output["messages"][-1]
        print(output["messages"])
        
        if isinstance(last_message, ToolMessage):
            content = last_message.content
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
                if hasattr(last_message, 'tool_calls'):
                    tool_calls = last_message.tool_calls
                    for tool_call in tool_calls:
                        if tool_call['name'] == "execute_shell_command_in_env":
                            msg=f"Running {tool_call['args']['command']}"
                            st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )   
                        if tool_call['name'] == "execute_python_code_in_env":
                            code = tool_call['args']['code']
                            st.code(code, language="python")

            except Exception as e:
                raise e
                breakpoint()

    st.session_state.messages.append({"role": "user", "content": prompt})
    if content:
        st.session_state.messages.append({"role": "assistant", "content": content})