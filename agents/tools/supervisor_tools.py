from datetime import datetime

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage
from langgraph.types import Send
from langgraph.graph import MessagesState

from typing import Annotated

def newest_user_text(messages: list) -> str | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            # HumanMessage.content is str *or* rich-text list
            return m.content if isinstance(m.content, str) else m.content[0]["text"]
        if isinstance(m, dict) and m.get("role") == "user":
            return m["content"]
    return None

def create_pricing_handoff_tool(agent_name: str):
    """Return a hand-off tool that injects ``question`` into the sub-graph state."""
    @tool(f"transfer_to_{agent_name}",
          description=f"Handoff to {agent_name} and provide the user question about flats' details for a specific building complex.")
    def _handoff_tool(
        state: Annotated[dict, InjectedState],
        #tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        question = newest_user_text(state.get("messages", []))
        if question is None:
            question = ""   # avoid KeyError in pricing graph; you may prefer to raise
        # record the hand-off in the chat history (optional)
        #state["messages"].append(
        #    ToolMessage(
        #        name=f"transfer_to_{agent_name}",
        #        content=f"Handoff → {agent_name}",
        #        tool_call_id=tool_call_id,
        #        response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name},
        #    )
        #)
        agent_input = {"question": question}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )
    _handoff_tool.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return _handoff_tool


def create_handoff_tool_no_history(agent_name: str, agent_purpose: str | None = None):
    name = f"transfer_to_{agent_name}"
    @tool(name,
          description=f"Handoff to {agent_name} to {agent_purpose or "execute agent specific tasks."} .")
    def _handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        task: Annotated[str, "Task delegated to agent"],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        question = newest_user_text(state.get("messages", []))
        content = f"User request: {question}\nPerform this task to address user request: {task}\nCurrent user datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        agent_input = {"messages": HumanMessage(content=content)}
        return Command(
            goto=[Send(agent_name, agent_input)],
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT
        )
    _handoff_tool.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return _handoff_tool


def create_handoff_tool_with_summary(agent_name: str, agent_purpose: str | None = None):
    name = f"transfer_to_{agent_name}"
    @tool(name,
          description=f"Handoff to {agent_name} to {agent_purpose or "execute agent specific tasks."} .")
    def _handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        task: Annotated[str, "Task delegated to agent"],
        summary: Annotated[str, "Summary of a chat with client. Shall include information about client, it's family, reason for purchasing flat, building complex, financial conditions client is interested in, number of rooms, budget (if provided, optional)"],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        question = newest_user_text(state.get("messages", []))
        content = f"User request: {question}\nPerform this task to address user request: {task}\n Chat summary: {summary}"
        agent_input = {"messages": HumanMessage(content=content)}
        return Command(
            goto=[Send(agent_name, agent_input)],
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT
        )
    _handoff_tool.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return _handoff_tool