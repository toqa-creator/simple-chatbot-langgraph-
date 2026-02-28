import os
import time
import operator
from datetime import datetime
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph_supervisor import create_supervisor

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")


router_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=api_key,
)

worker_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
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
    """Divide a by b. Raises error for division by zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@tool
def get_current_time_and_date() -> str:
    """Return the current local time and date."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M:%S")

CALC_TOOLS = [add, subtract, multiply, divide]
CALC_TOOLS_BY_NAME = {t.name: t for t in CALC_TOOLS}
calc_model_with_tools = worker_model.bind_tools(CALC_TOOLS)


class CalcState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]


def calc_llm_call(state: CalcState) -> CalcState:
    system = SystemMessage(
        content=(
            "You are a calculator agent.\n"
            "You MUST use the provided tools to compute results.\n"
            "Respect operator precedence.\n"
            "Return ONLY the final numeric answer as plain text."
        )
    )
    resp = calc_model_with_tools.invoke([system] + state.get("messages", []))
    return {"messages": [resp]}


def calc_tool_node(state: CalcState) -> CalcState:
    last = state["messages"][-1]
    tool_msgs: list[ToolMessage] = []

    for call in getattr(last, "tool_calls", []) or []:
        name = call.get("name")
        args = call.get("args", {}) or {}
        tool_fn = CALC_TOOLS_BY_NAME.get(name)

        if not tool_fn:
            tool_msgs.append(
                ToolMessage(
                    content=f"Unknown tool: {name}",
                    tool_call_id=call.get("id", ""),
                )
            )
            continue

        try:
            obs = tool_fn.invoke(args)
            tool_msgs.append(
                ToolMessage(
                    content=str(obs),
                    tool_call_id=call.get("id", ""),
                )
            )
        except Exception as e:
            tool_msgs.append(
                ToolMessage(
                    content=f"Tool error: {e}",
                    tool_call_id=call.get("id", ""),
                )
            )

    return {"messages": tool_msgs}


def calc_should_continue(state: CalcState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and len(last.tool_calls) > 0:
        return "tool_node"
    return END


calc_graph = StateGraph(CalcState)
calc_graph.add_node("llm_call", calc_llm_call)
calc_graph.add_node("tool_node", calc_tool_node)

calc_graph.add_edge(START, "llm_call")
calc_graph.add_conditional_edges("llm_call", calc_should_continue, ["tool_node", END])
calc_graph.add_edge("tool_node", "llm_call")

calculator_agent = calc_graph.compile()
calculator_agent.name = "calculator"

class TimeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]


def time_node(state: TimeState) -> TimeState:
    _ = state
    now_str = get_current_time_and_date.invoke({})
    return {"messages": [AIMessage(content=str(now_str))]}


time_graph = StateGraph(TimeState)
time_graph.add_node("time", time_node)
time_graph.add_edge(START, "time")
time_graph.add_edge("time", END)

time_agent = time_graph.compile()
time_agent.name = "time"

workflow = create_supervisor(
    [calculator_agent, time_agent],
    model=router_model,
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- calculator: handles math calculations\n"
        "- time: handles current time/date questions\n\n"
        "Rules:\n"
        "1) Always route the request to ONE best agent.\n"
        "2) After the agent returns, you MUST answer the user with the agent's final result.\n"
        "3) Do NOT add extra commentary. Do NOT call any other tools after you have the answer.\n"
        "4) If the user greeting only (hi/hello), reply with a short greeting.\n\n"
        "If math -> calculator.\n"
        "If time/date -> time.\n"
        "If unclear, ask a short clarifying question."
    ),
)

graph = workflow.compile()

def invoke_with_retry(app, payload, retries: int = 6, start_delay: float = 1.0):
    delay = start_delay
    last_err = None
    for _ in range(retries):
        try:
            return app.invoke(payload)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError(f"Rate limit persists after retries. Last error: {last_err}")


if __name__ == "__main__":
    res1 = invoke_with_retry(graph, {"messages": [HumanMessage(content="Calculate 2 + 2 * 4 = ?")]})
    print("Math:", res1["messages"][-1].content)

    res2 = invoke_with_retry(graph, {"messages": [HumanMessage(content="What time is it now?")]})
    print("Time:", res2["messages"][-1].content)