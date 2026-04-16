import os
from utils.langfuse import run_llm_call
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.analyzer import conversation_analyzer_agent
from agents.locations_agent import analyze_location_data


#TODO finish this prompt
ORCHESTRATOR_PROMPT = """
You are the Chief Fraud Operations Officer. Your goal is to oversee a multi-stage investigation into potential user account compromise. You have access to specialized agents (Tools) that analyze communication patterns and geospatial movements. 
Your job is to determine if a transaction is fraudolent.
This can happen because a user is being targeted by "Blended Fraud"—a sophisticated attack where digital deception (SMS/Email) and physical anomalies (GPS) converge, resulting into a fraudolent transaction.

Stage 1: Linguistic Intake: 
Call the conversation_analyzer_agent with any provided SMS or Email data. Look for high-probability fraud or "Low-Observable" indicators (social engineering, brand mimicry).

Stage 2: Contextual Correlation: 
Take the findings from the communication analysis (the fraud_detail and confidence_score) and use them as additional_point_of_concerns when calling the analyze_location_data tool.

Stage 3: Physical Verification: 
Use the location tool to check if the user's physical presence at the time of the message/transaction supports or contradicts the fraud hypothesis.

Stage 4: Transaction Verification: 
"""

orchestrator_llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
)

#TODO: need to write a transaction agent
orchestrator_agent = create_agent(
    model = orchestrator_llm,
    system_prompt = ORCHESTRATOR_PROMPT,
    tools = [conversation_analyzer_agent, analyze_location_data]
)

def call_orchestrator(session_id) -> str:
    message = {'messages':[HumanMessage(f"The session_id is {session_id}. Use it to when calling the agent tools.")]}
    response = run_llm_call(session_id, orchestrator_agent, message)
    return response["messages"][-1].content