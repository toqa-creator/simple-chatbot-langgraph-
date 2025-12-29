import os
import re
import json
import operator
import ast
from datetime import datetime

from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, START, END



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=api_key,
)



def extract_text_from_message(msg: AnyMessage) -> str:
    """
    Studio sometimes sends content as list parts or dict-like messages.
    This returns the text in any form.
    """
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if t:
                    texts.append(t)
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts).strip()

    if isinstance(content, str):
        return content.strip()

    return ""


def extract_math_expression(user_text: str) -> str:
    """
    Extracts a math expression from Arabic/English text.
    Examples:
      " calc 2 + 3 * 4" -> "2 + 3 * 4"
      "2x4 + 1" -> "2*4 + 1"
    """
    if not user_text:
        return ""

    text = user_text

    
    text = text.replace("×", "*")
    text = text.replace("÷", "/")

    
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", text)

    
    matches = re.findall(r"[0-9\.\+\-\*/\(\)\s]+", text)
    if not matches:
        return ""

    expr = max(matches, key=len).strip()
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr



ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr: str) -> float:
    """
    Supports + - * / and parentheses safely using AST.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)

        
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            return float(n.n)

        if isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Div) and right == 0:
                raise ValueError("Division by zero is not allowed.")
            return ALLOWED_BINOPS[type(n.op)](left, right)

        if isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_UNARYOPS:
            return ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand))

        raise ValueError("Unsupported expression")

    return _eval(node)


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(x)


def get_current_time_and_date() -> str:
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M:%S")



class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    next: str
    routed_input: str



def calculator_node(state: AgentState) -> AgentState:
    user_text = (state.get("routed_input") or "").strip()
    expr = extract_math_expression(user_text)

    if not expr:
        return {"messages": [AIMessage(content="Write a clear equation like: 2 + 3 * 4")]}

    try:
        result = safe_eval_expr(expr)
        return {"messages": [AIMessage(content=format_number(result))]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"There's an error in the equation:{e}")]}  


calc_builder = StateGraph(AgentState)
calc_builder.add_node("calc", calculator_node)
calc_builder.add_edge(START, "calc")
calc_builder.add_edge("calc", END)
calculator_agent = calc_builder.compile()



def time_node(state: AgentState) -> AgentState:
    return {"messages": [AIMessage(content=get_current_time_and_date())]}


time_builder = StateGraph(AgentState)
time_builder.add_node("time", time_node)
time_builder.add_edge(START, "time")
time_builder.add_edge("time", END)
time_agent = time_builder.compile()



def supervisor_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    user_text = extract_text_from_message(last_msg)

    system = SystemMessage(content=(
        "You are a supervisor router for a LangGraph app.\n"
        "You must choose the best next agent for the user's last message.\n"
        "Allowed agents: calculator, time.\n\n"
        "Rules:\n"
        "- If the user asks about current time/date -> time\n"
        "- If the user asks to calculate / evaluate an expression -> calculator\n"
        "- If there is a clear math expression inside the text -> calculator\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\"next\":\"calculator\",\"routed_input\":\"...\"}\n"
        "Where routed_input is the cleaned user message to pass to the chosen agent."
    ))

    resp = model.invoke([system, HumanMessage(content=user_text)])
    raw = (resp.content or "").strip()

   
    try:
        data = json.loads(raw)
    except Exception:
        
        expr = extract_math_expression(user_text)
        nxt = "calculator" if expr else "time"
        data = {"next": nxt, "routed_input": user_text}

    nxt = data.get("next", "calculator")
    if nxt not in {"calculator", "time"}:
        nxt = "calculator"

    routed_input = data.get("routed_input", user_text)
    if not isinstance(routed_input, str):
        routed_input = user_text

    return {"next": nxt, "routed_input": routed_input}


def decide_next_agent(state: AgentState):
    return state.get("next", "calculator")



manager_builder = StateGraph(AgentState)

manager_builder.add_node("supervisor", supervisor_node)
manager_builder.add_node("calculator", calculator_agent)
manager_builder.add_node("time", time_agent)

manager_builder.add_edge(START, "supervisor")

manager_builder.add_conditional_edges(
    "supervisor",
    decide_next_agent,
    ["calculator", "time"],
)

manager_builder.add_edge("calculator", END)
manager_builder.add_edge("time", END)

manager_agent = manager_builder.compile()



def manager_chat_loop() -> None:
    print("Manager Agent (Supervisor + Calculator + Time/Date via LangGraph)")
    print("Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Bye")
            break

        state = manager_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        ai_messages = [m for m in state["messages"] if getattr(m, "type", None) == "ai"]
        print("Bot:", ai_messages[-1].content if ai_messages else "(no response)")


if __name__ == "__main__":
    manager_chat_loop()
