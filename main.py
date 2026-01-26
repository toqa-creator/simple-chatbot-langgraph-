import os
import re
import json
import operator
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END



load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

router_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=api_key,
)

calc_model = ChatGoogleGenerativeAI(
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


TOOLS = [add, subtract, multiply, divide]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
calc_model_with_tools = calc_model.bind_tools(TOOLS)



def get_text(msg) -> str:
    """Works with BaseMessage objects AND dict messages from Studio."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()

    return ""


def normalize_messages(messages: list) -> list[AnyMessage]:
    """Convert Studio dict messages into LangChain Message objects."""
    normalized: list[AnyMessage] = []

    for m in messages or []:
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage, SystemMessage)):
            normalized.append(m)
            continue

        if isinstance(m, dict):
            t = (m.get("type") or m.get("role") or "").lower()
            content = m.get("content", "")

            if t in {"human", "user"}:
                normalized.append(HumanMessage(content=content))
            elif t in {"ai", "assistant"}:
                normalized.append(AIMessage(content=content))
            elif t == "tool":
                normalized.append(
                    ToolMessage(
                        content=str(content),
                        tool_call_id=m.get("tool_call_id", "") or m.get("id", "")
                    )
                )
            elif t == "system":
                normalized.append(SystemMessage(content=content))
            else:
                normalized.append(HumanMessage(content=str(content)))
        else:
            normalized.append(HumanMessage(content=str(m)))

    return normalized


def get_current_time_and_date() -> str:
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M:%S")


def looks_like_time_question(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in [
        "time", "date", "today", "now",
        "الوقت", "الساعة", "التاريخ", "النهارده", "الآن", "دلوقتي"
    ])


def looks_like_math(text: str) -> bool:
    return bool(re.search(r"\d", text)) and bool(re.search(r"[\+\-\*/×÷]", text))



class CalcState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]


def calc_llm_call(state: CalcState) -> CalcState:
    system = SystemMessage(content=(
        "You are a calculator assistant.\n"
        "Use ONLY the provided tools (add, subtract, multiply, divide).\n"
        "Respect operator precedence.\n"
        "Return a clear final numeric answer only."
    ))

    msgs = normalize_messages(state.get("messages", []))
    resp = calc_model_with_tools.invoke([system] + msgs)
    return {"messages": [resp]}


def calc_tool_node(state: CalcState) -> CalcState:
    last = normalize_messages(state.get("messages", []))[-1]
    tool_msgs: list[ToolMessage] = []

    for call in getattr(last, "tool_calls", []) or []:
        name = call.get("name")
        args = call.get("args", {}) or {}
        tool_fn = TOOLS_BY_NAME.get(name)

        if not tool_fn:
            tool_msgs.append(ToolMessage(
                content=f"Unknown tool: {name}",
                tool_call_id=call.get("id", "")
            ))
            continue

        try:
            obs = tool_fn.invoke(args)
            tool_msgs.append(ToolMessage(
                content=str(obs),
                tool_call_id=call.get("id", "")
            ))
        except Exception as e:
            tool_msgs.append(ToolMessage(
                content=f"Tool error: {e}",
                tool_call_id=call.get("id", "")
            ))

    return {"messages": tool_msgs}


def calc_should_continue(state: CalcState) -> str:
    last = normalize_messages(state.get("messages", []))[-1]
    if getattr(last, "tool_calls", None) and len(last.tool_calls) > 0:
        return "tool_node"
    return END


calc_builder = StateGraph(CalcState)
calc_builder.add_node("llm_call", calc_llm_call)
calc_builder.add_node("tool_node", calc_tool_node)
calc_builder.add_edge(START, "llm_call")
calc_builder.add_conditional_edges("llm_call", calc_should_continue, ["tool_node", END])
calc_builder.add_edge("tool_node", "llm_call")
calculator_agent = calc_builder.compile()



class TimeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]


def time_node(state: TimeState) -> TimeState:
    return {"messages": [AIMessage(content=get_current_time_and_date())]}


time_builder = StateGraph(TimeState)
time_builder.add_node("time", time_node)
time_builder.add_edge(START, "time")
time_builder.add_edge("time", END)
time_agent = time_builder.compile()



class ManagerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    next: str
    routed_input: str


def supervisor_node(state: ManagerState) -> ManagerState:
    
    msgs = normalize_messages(state.get("messages", []))
    if not msgs:
        return {"next": "calculator", "routed_input": "2+2"}

    last_msg = msgs[-1]
    user_text = get_text(last_msg)

    
    if not user_text:
        raw_last = state.get("messages", [])[-1] if state.get("messages") else {}
        if isinstance(raw_last, dict):
            user_text = (raw_last.get("content") or "").strip()

    if not user_text:
        return {"next": "calculator", "routed_input": "2+2"}

    
    if looks_like_time_question(user_text) and not looks_like_math(user_text):
        return {"next": "time", "routed_input": user_text}
    if looks_like_math(user_text):
        return {"next": "calculator", "routed_input": user_text}

    system = SystemMessage(content=(
        "You are a supervisor router for a LangGraph app.\n"
        "Choose the best next agent for the user's last message.\n"
        "Allowed agents: calculator, time.\n\n"
        "Rules:\n"
        "- If user asks about current time/date -> time\n"
        "- If user asks to calculate/evaluate -> calculator\n\n"
        "Return ONLY valid JSON exactly like:\n"
        "{\"next\":\"calculator\",\"routed_input\":\"...\"}"
    ))

    resp = router_model.invoke([system, HumanMessage(content=user_text)])
    raw = (resp.content or "").strip()

    try:
        data = json.loads(raw)
        nxt = data.get("next", "calculator")
        routed_input = data.get("routed_input", user_text)
    except Exception:
        nxt = "calculator"
        routed_input = user_text

    if nxt not in {"calculator", "time"}:
        nxt = "calculator"
    if not isinstance(routed_input, str):
        routed_input = user_text

    return {"next": nxt, "routed_input": routed_input}


def decide_next(state: ManagerState) -> Literal["calculator", "time"]:
    return "time" if state.get("next") == "time" else "calculator"


manager_builder = StateGraph(ManagerState)
manager_builder.add_node("supervisor", supervisor_node)
manager_builder.add_node("calculator", calculator_agent)
manager_builder.add_node("time", time_agent)

manager_builder.add_edge(START, "supervisor")
manager_builder.add_conditional_edges("supervisor", decide_next, ["calculator", "time"])
manager_builder.add_edge("calculator", END)
manager_builder.add_edge("time", END)


graph = manager_builder.compile()
