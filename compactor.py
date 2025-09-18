"""
Context Compactor for LangGraph React Agents

This module provides intelligent context length management by compacting tool responses
and long AI messages while preserving user inputs and recent conversation context.
"""

import streamlit as st
from typing import Annotated, Sequence
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages


# Configuration constants
DEFAULT_MAX_TOKENS = 20000
DEFAULT_COMPACTION_THRESHOLD = 0.8
DEFAULT_PRESERVE_RECENT = 3
DEFAULT_COMPACTION_MODEL = "gpt-5-mini"


class ContextAwareAgentState(AgentState):
    """Enhanced agent state with context management."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    context_tokens: int = 0
    compaction_count: int = 0


class ContextCompactor:
    """Handles context compaction for LangGraph React agents."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD,
        compaction_model: str = DEFAULT_COMPACTION_MODEL,
        preserve_recent_messages: int = DEFAULT_PRESERVE_RECENT,
        openai_api_key: str = None,
    ):
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self.trigger_tokens = int(max_tokens * compaction_threshold)
        self.preserve_recent_messages = preserve_recent_messages

        # Initialize compaction LLM
        api_key = openai_api_key or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for context compaction")

        self.compaction_llm = ChatOpenAI(
            model=compaction_model,
            api_key=api_key,
            thinking="minimal",
            max_tokens=1000,
        )

    def count_message_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """Count tokens in message sequence."""
        return count_tokens_approximately(messages)

    def compact_messages(
        self, messages: Sequence[BaseMessage]
    ) -> Sequence[BaseMessage]:
        """Compact messages while preserving recent context and user messages."""
        total_tokens = self.count_message_tokens(messages)

        if total_tokens <= self.max_tokens:
            return messages

        # Debug info for tool message handling
        tool_msgs = [msg for msg in messages if isinstance(msg, ToolMessage)]
        tool_count = len(tool_msgs)
        error_tools = len(
            [msg for msg in tool_msgs if getattr(msg, "status", "success") == "error"]
        )
        ai_with_tools = len(
            [
                msg
                for msg in messages
                if isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ]
        )

        st.info(
            f"🔄 Context compaction needed: {total_tokens:,} -> target: {self.max_tokens:,} tokens\n"
            f"📊 Messages: {len(messages)} total | {tool_count} tool responses ({error_tools} errors) | {ai_with_tools} AI with tool calls"
        )

        # Separate system messages (always preserve)
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Always preserve recent messages
        recent_msgs = (
            other_msgs[-self.preserve_recent_messages * 2 :]
            if len(other_msgs) > self.preserve_recent_messages * 2
            else other_msgs
        )
        older_msgs = (
            other_msgs[: -self.preserve_recent_messages * 2]
            if len(other_msgs) > self.preserve_recent_messages * 2
            else []
        )

        # Compact older messages
        compacted_older = self._compact_older_messages(older_msgs)

        # Combine: system + compacted_older + recent
        final_messages = system_msgs + compacted_older + recent_msgs

        final_tokens = self.count_message_tokens(final_messages)
        final_tool_msgs = [
            msg for msg in final_messages if isinstance(msg, ToolMessage)
        ]
        final_tool_count = len(final_tool_msgs)
        final_error_tools = len(
            [
                msg
                for msg in final_tool_msgs
                if getattr(msg, "status", "success") == "error"
            ]
        )

        st.success(
            f"✅ Context compacted to {final_tokens:,} tokens ({len(final_messages)} messages)\n"
            f"🛠️ Tool responses: {tool_count} -> {final_tool_count} ({final_error_tools} errors preserved)"
        )

        return final_messages

    def _compact_older_messages(
        self, messages: Sequence[BaseMessage]
    ) -> list[BaseMessage]:
        """Compact older messages by summarizing tool calls and long responses."""
        if not messages:
            return []

        # Group messages by conversation turns (user -> ai -> tools -> ai pattern)
        conversation_turns = self._group_into_turns(messages)
        compacted_turns = []

        for turn in conversation_turns:
            compacted_turn = self._compact_conversation_turn(turn)
            compacted_turns.extend(compacted_turn)

        return compacted_turns

    def _group_into_turns(
        self, messages: Sequence[BaseMessage]
    ) -> list[list[BaseMessage]]:
        """Group messages into conversation turns, respecting tool call flows."""
        turns = []
        current_turn = []

        for i, msg in enumerate(messages):
            current_turn.append(msg)

            # Complex logic to handle tool call patterns:
            # User -> AI (with tool calls) -> Tool responses -> AI (final response)
            if isinstance(msg, AIMessage):
                # Check if this AI message has tool calls
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls

                if has_tool_calls:
                    # Look ahead for corresponding tool responses
                    remaining_msgs = messages[i + 1 :]
                    tool_responses_ahead = []

                    # Collect tool responses that match this AI message's tool calls
                    tool_call_ids = [call.get("id", "") for call in msg.tool_calls]

                    for next_msg in remaining_msgs:
                        if isinstance(next_msg, ToolMessage):
                            if getattr(next_msg, "tool_call_id", "") in tool_call_ids:
                                tool_responses_ahead.append(next_msg)
                        elif isinstance(next_msg, AIMessage):
                            # Stop looking when we hit the next AI message
                            break
                        elif isinstance(next_msg, HumanMessage):
                            # Stop if user interrupts
                            break

                    # Don't end turn yet if we expect tool responses
                    if tool_responses_ahead:
                        continue
                else:
                    # AI message without tool calls - check if we should end the turn
                    # Look ahead to see if there are more messages in this conversation
                    next_msgs = messages[i + 1 : i + 3]  # Look ahead 2 messages
                    has_immediate_tool_or_user = any(
                        isinstance(next_msg, (ToolMessage, HumanMessage))
                        for next_msg in next_msgs
                    )

                    if not has_immediate_tool_or_user:
                        # End the turn - this seems like a final AI response
                        turns.append(current_turn)
                        current_turn = []

            elif isinstance(msg, ToolMessage):
                # Tool message - check if there are more tool responses coming
                # or if we should wait for the AI's final response
                remaining_msgs = messages[i + 1 :]

                # Look for the next non-tool message
                next_non_tool = None
                for next_msg in remaining_msgs[:5]:  # Check next 5 messages
                    if not isinstance(next_msg, ToolMessage):
                        next_non_tool = next_msg
                        break

                # If the next non-tool message is AI, continue the turn
                # If it's a user message, end the turn
                if isinstance(next_non_tool, HumanMessage):
                    turns.append(current_turn)
                    current_turn = []

            elif isinstance(msg, HumanMessage):
                # User messages typically start new turns, but we already added this message
                # so we continue collecting until we find the response pattern
                continue

        # Add any remaining messages as the final turn
        if current_turn:
            turns.append(current_turn)

        return turns

    def _compact_conversation_turn(self, turn: list[BaseMessage]) -> list[BaseMessage]:
        """Compact a single conversation turn, preserving tool call relationships."""
        if len(turn) <= 2:  # Short turn, keep as is
            return turn

        # Separate message types
        user_msgs = [msg for msg in turn if isinstance(msg, HumanMessage)]
        tool_msgs = [msg for msg in turn if isinstance(msg, ToolMessage)]
        ai_msgs = [msg for msg in turn if isinstance(msg, AIMessage)]

        compacted_turn = []

        # Always preserve user messages
        compacted_turn.extend(user_msgs)

        # Handle AI messages with potential tool calls
        for ai_msg in ai_msgs:
            # Check if this AI message has tool calls
            has_tool_calls = hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls

            if has_tool_calls:
                # Keep AI message with tool calls (important for understanding what was requested)
                if len(ai_msg.content) > 2000:
                    # Summarize very long AI messages but keep tool call info
                    summarized_ai = self._summarize_ai_message(ai_msg)
                    compacted_turn.append(summarized_ai)
                else:
                    compacted_turn.append(ai_msg)

                # Find and handle corresponding tool responses
                ai_tool_call_ids = (
                    [call.get("id", "") for call in ai_msg.tool_calls]
                    if ai_msg.tool_calls
                    else []
                )
                related_tool_msgs = [
                    msg
                    for msg in tool_msgs
                    if getattr(msg, "tool_call_id", "") in ai_tool_call_ids
                ]

                if related_tool_msgs:
                    # Check if any tool messages are short and successful
                    short_successful_tools = [
                        msg
                        for msg in related_tool_msgs
                        if len(str(getattr(msg, "content", ""))) < 1500
                        and getattr(msg, "status", "success") == "success"
                    ]

                    # Keep short successful tools, summarize others
                    if len(related_tool_msgs) == 1 and short_successful_tools:
                        compacted_turn.extend(related_tool_msgs)
                    else:
                        # Summarize multiple or long tool responses
                        summarized_tool = self._summarize_tool_messages(
                            related_tool_msgs
                        )
                        compacted_turn.append(summarized_tool)

                    # Remove processed tool messages from the list
                    tool_msgs = [
                        msg for msg in tool_msgs if msg not in related_tool_msgs
                    ]
            else:
                # Regular AI message without tool calls
                if len(ai_msg.content) > 1500:
                    summarized_ai = self._summarize_ai_message(ai_msg)
                    compacted_turn.append(summarized_ai)
                else:
                    compacted_turn.append(ai_msg)

        # Handle any remaining tool messages (orphaned or from previous turns)
        if tool_msgs:
            # Check for short successful tools
            short_tools = [
                msg
                for msg in tool_msgs
                if len(str(getattr(msg, "content", ""))) < 1000
                and getattr(msg, "status", "success") == "success"
            ]

            if len(tool_msgs) == 1 and short_tools:
                compacted_turn.extend(tool_msgs)
            else:
                summarized_remaining = self._summarize_tool_messages(tool_msgs)
                compacted_turn.append(summarized_remaining)

        return compacted_turn

    def _validate_tool_message(self, msg: ToolMessage) -> dict:
        """Validate and extract ToolMessage fields safely based on LangChain structure."""
        content = getattr(msg, "content", "")

        # Handle content that can be str or list[Union[str, dict]]
        if isinstance(content, list):
            # Flatten list content for better summarization
            content_str = ""
            for item in content:
                if isinstance(item, dict):
                    content_str += f"{item}\n"
                else:
                    content_str += f"{str(item)}\n"
            content = content_str
        else:
            content = str(content)

        return {
            "content": content,
            "tool_call_id": getattr(msg, "tool_call_id", ""),  # Required field
            "type": getattr(msg, "type", "tool"),
            "artifact": getattr(msg, "artifact", None),  # Additional data not for model
            "status": getattr(msg, "status", "success"),  # success/error status
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            "response_metadata": getattr(msg, "response_metadata", {}),
        }

    def _summarize_tool_messages(self, tool_msgs: list[ToolMessage]) -> ToolMessage:
        """Summarize multiple tool messages with proper LangChain ToolMessage field handling."""
        try:
            # Combine tool outputs for summarization
            combined_content = ""
            tool_call_ids = []
            error_count = 0
            success_count = 0

            for i, msg in enumerate(tool_msgs):
                # Safely extract ToolMessage fields
                msg_data = self._validate_tool_message(msg)

                tool_call_ids.append(msg_data["tool_call_id"])

                # Track success/error status
                if msg_data["status"] == "error":
                    error_count += 1
                else:
                    success_count += 1

                # Truncate very long tool responses before summarization
                content = msg_data["content"]
                content_preview = content[:1200] if len(content) > 1200 else content

                combined_content += (
                    f"\n--- Tool Call #{i+1} (ID: {msg_data['tool_call_id']}) ---\n"
                )
                combined_content += f"Status: {msg_data['status']}\n"
                combined_content += f"Content: {content_preview}\n"

                if len(content) > 1200:
                    combined_content += (
                        f"[... truncated from {len(content)} characters]\n"
                    )

                # Include artifact information if present
                if msg_data["artifact"]:
                    artifact_summary = str(msg_data["artifact"])[:200]
                    combined_content += f"Artifact: {artifact_summary}{'...' if len(str(msg_data['artifact'])) > 200 else ''}\n"

                # Include any error information from metadata
                if msg_data["response_metadata"]:
                    combined_content += f"Metadata: {msg_data['response_metadata']}\n"

            # Status summary for the prompt
            status_summary = (
                f"({success_count} successful, {error_count} errors)"
                if error_count > 0
                else f"(all {success_count} successful)"
            )

            summary_prompt = f"""Summarize these tool execution results {status_summary} concisely while preserving essential information:

{combined_content}

Instructions:
- Keep key data, findings, and technical details
- Preserve important numbers, URLs, file paths, or configuration details  
- **ALWAYS mention any errors or failures clearly**
- If tools returned structured data (JSON, CSV), mention the data structure
- Include artifact information if significant
- Summarize in under 400 words
- Maintain the overall success/failure context"""

            summary_response = self.compaction_llm.invoke(
                [HumanMessage(content=summary_prompt)]
            )

            # Create summarized ToolMessage with proper structure
            # Use the first tool_call_id as the primary ID
            primary_tool_call_id = (
                tool_call_ids[0] if tool_call_ids else "compacted_tools"
            )

            # Determine overall status - if any failed, mark as error
            overall_status = "error" if error_count > 0 else "success"

            return ToolMessage(
                content=f"[COMPACTED {len(tool_msgs)} tool calls {status_summary}]\n\n{summary_response.content}",
                tool_call_id=primary_tool_call_id,
                status=overall_status,
            )

        except Exception as e:
            # Fallback: create a simple summary with proper ToolMessage structure
            tool_call_ids = []
            total_content_length = 0
            error_tools = 0

            for msg in tool_msgs:
                msg_data = self._validate_tool_message(msg)
                tool_call_ids.append(msg_data["tool_call_id"])
                total_content_length += len(msg_data["content"])
                if msg_data["status"] == "error":
                    error_tools += 1

            fallback_content = f"[COMPACTED {len(tool_msgs)} tool responses]\n"
            fallback_content += (
                f"Tool calls: {len(tool_msgs)} total, {error_tools} errors\n"
            )
            fallback_content += (
                f"Total original content: ~{total_content_length:,} characters\n"
            )
            fallback_content += f"Summarization failed: {str(e)}\n"
            fallback_content += (
                "Note: Tool responses were compacted due to length constraints."
            )

            return ToolMessage(
                content=fallback_content,
                tool_call_id=(
                    tool_call_ids[0] if tool_call_ids else "compacted_fallback"
                ),
                status="error" if error_tools > 0 else "success",
            )

    def _summarize_ai_message(self, ai_msg: AIMessage) -> AIMessage:
        """Summarize a long AI message."""
        try:
            summary_prompt = f"""Summarize this AI assistant response, preserving key information, decisions, and any code or technical details:

{ai_msg.content}

Provide a concise summary in under 200 words that maintains the essential points."""

            summary_response = self.compaction_llm.invoke(
                [HumanMessage(content=summary_prompt)]
            )

            return AIMessage(
                content=f"[SUMMARIZED AI RESPONSE]\n{summary_response.content}"
            )

        except Exception:
            # Fallback to truncation
            truncated = ai_msg.content[:1000] + "...[TRUNCATED]"
            return AIMessage(content=truncated)


def create_context_management_node(compactor: ContextCompactor):
    """Factory function to create a context management node with the given compactor."""

    def context_management_node(
        state: ContextAwareAgentState,
    ) -> ContextAwareAgentState:
        """Node that handles context compaction before processing."""
        messages = state["messages"]
        current_cumulative_tokens = state.get("context_tokens", 0)

        # Count tokens in current message batch
        batch_tokens = compactor.count_message_tokens(messages)
        total_tokens = current_cumulative_tokens + batch_tokens

        # Check if we need to show warning (approaching threshold)
        if (
            total_tokens > compactor.trigger_tokens
            and current_cumulative_tokens <= compactor.trigger_tokens
        ):
            st.warning(
                f"⚠️ Context approaching limit: {total_tokens:,}/{compactor.max_tokens:,} tokens"
            )

        # Compact if exceeded max tokens
        if total_tokens > compactor.max_tokens:
            st.info(f"🔄 Context compaction triggered: {total_tokens:,} tokens")
            compacted_messages = compactor.compact_messages(messages)
            final_tokens = compactor.count_message_tokens(compacted_messages)

            st.success(f"✅ Context compacted to {final_tokens:,} tokens")

            return {
                "messages": compacted_messages,
                "context_tokens": final_tokens,  # Reset to compacted size
                "compaction_count": state.get("compaction_count", 0) + 1,
            }

        # No compaction needed
        return {
            "messages": messages,
            "context_tokens": total_tokens,  # Accumulate tokens
            "compaction_count": state.get("compaction_count", 0),
        }

    return context_management_node


# Convenience function for easy integration
def create_context_aware_agent(
    base_agent,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    compaction_model: str = DEFAULT_COMPACTION_MODEL,
    openai_api_key: str = None,
):
    """
    Create a context-aware version of any LangGraph agent.

    Args:
        base_agent: The base LangGraph agent (e.g., create_react_agent result)
        max_tokens: Maximum context tokens before compaction
        compaction_threshold: Trigger compaction at this ratio of max_tokens
        compaction_model: OpenAI model to use for summarization
        openai_api_key: OpenAI API key (uses st.secrets if None)

    Returns:
        Compiled LangGraph workflow with context management
    """
    from langgraph.graph import StateGraph, START, END

    # Initialize compactor
    compactor = ContextCompactor(
        max_tokens=max_tokens,
        compaction_threshold=compaction_threshold,
        compaction_model=compaction_model,
        openai_api_key=openai_api_key,
    )

    # Create context management node
    context_node = create_context_management_node(compactor)

    # Create new workflow with context management
    workflow = StateGraph(ContextAwareAgentState)
    workflow.add_node("context_manager", context_node)
    workflow.add_node("base_agent", base_agent)

    # Set up flow: context management -> base agent
    workflow.add_edge(START, "context_manager")
    workflow.add_edge("context_manager", "base_agent")
    workflow.add_edge("base_agent", END)

    return workflow.compile()
