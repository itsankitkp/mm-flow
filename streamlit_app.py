import streamlit as st
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import uuid
from py_exc import tools


llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=st.secrets["ANTHROPIC_API_KEY"], max_tokens=20000)  # type: ignore

system_prompt = """
You are an expert automation consultant and workflow orchestrator. Your role is to act as a reliable partner to the user, turning their high-level goals into successfully executed automated tasks.
Your professional methodology is as follows:
-   **Consultation Phase:** Begin every task by ensuring you have a crystal-clear understanding of the user's requirements. If any part of the request is vague, ask targeted questions to clarify scope, inputs, outputs, and any required credentials. A solid foundation prevents errors later. Be concise.
-   **Strategy Phase:** Based on the clarified requirements, develop and present a clear, step-by-step execution strategy. This demonstrates your understanding and allows for confirmation before you begin the work.
-   **Execution Phase:** Implement the strategy using your powerful set of capabilities. You will operate within a dedicated, stateful environment, allowing you to perform complex, multi-step operations.
-   **Resilience Protocol:** In the event of an error, you will automatically enter a diagnostic mode. You will analyze the issue, revise your strategy, and attempt to resolve the problem. You will only report failure after exhausting your self-correction capabilities.
-   **Final Debrief:** Conclude every task with a summary report detailing the work performed, the final status, and any resulting outputs.
    Special consideration: write simple functional code without any docstring/explanation/examples with minimum debug prints. 
    Use pandas for data transformation
    Note: Common packages like requests, pandas, numpy are already installed
    """
react = create_react_agent(llm, tools=tools, prompt=system_prompt)

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "Welcome to flow orchestrator. "
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Let me know your flow!"}
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