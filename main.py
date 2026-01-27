import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
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



calculator_agent = create_react_agent(
    model=worker_model,
    name="calculator",
    tools=[add, subtract, multiply, divide],
    prompt=(
        "You are a calculator agent.\n"
        "You MUST use the provided tools to compute results.\n"
        "Respect operator precedence.\n"
        "Return ONLY the final numeric answer."
    ),
)

time_agent = create_react_agent(
    model=worker_model,
    name="time",
    tools=[get_current_time_and_date],
    prompt=(
        "You are a time/date agent.\n"
        "If user asks about current time/date, call get_current_time_and_date.\n"
        "Return the tool result only."
    ),
)



workflow = create_supervisor(
    [calculator_agent, time_agent],
    model=router_model,
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- calculator: handles math calculations\n"
        "- time: handles current time/date questions\n\n"
        "Route the user request to the best agent.\n"
        "If it is math -> calculator.\n"
        "If it is time/date -> time.\n"
        "If unclear, ask a short clarifying question."
    ),
)


graph = workflow.compile()
