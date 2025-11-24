import os
import operator
from typing import Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END




load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=api_key,
)



@tool
def add(a: float, b: float) -> float:
    """Add a and b."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


tools = [add, subtract, multiply, divide]
tools_by_name = {t.name: t for t in tools}


model_with_tools = model.bind_tools(tools)





class AgentState(TypedDict):
    
    messages: Annotated[list[AnyMessage], operator.add]


 

def llm_call(state: AgentState) -> AgentState:
    """Call the LLM, letting it decide whether to use tools."""
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a calculator assistant. "
                    "Use the tools (add, subtract, multiply, divide) to do math. "
                    "Always return the final numeric result clearly."
                )
            )
        ]
        + state["messages"]
    )
    return {"messages": [response]}




def tool_node(state: AgentState) -> AgentState:
    """Execute any tool calls from the last AI message."""
    last_msg = state["messages"][-1]
    results: list[ToolMessage] = []

    
    for tool_call in last_msg.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}





def should_continue(state: AgentState):
    """Route to tool_node if there are tool calls, otherwise end the graph."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END





builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")

builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)

builder.add_edge("tool_node", "llm_call")

agent = builder.compile()





def chat_loop() -> None:
    print("Simple LangGraph Calculator (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break

        state = agent.invoke({"messages": [HumanMessage(content=user_input)]})

        
        ai_messages = [m for m in state["messages"] if m.type == "ai"]
        if ai_messages:
            print("Bot:", ai_messages[-1].content)
        else:
            print("Bot: (no response)")


if __name__ == "__main__":
    chat_loop()
