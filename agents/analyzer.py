import os
from utils.langfuse import run_llm_call
from typing import List, Optional
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ANALYSIS_PROMPT = """
You are an Elite Fraud Detection Agent specializing in "Blended Social Engineering." 
Your task is to analyze communications (SMS/Email) to distinguish between legitimate community outreach and high-sophistication fraud.

Core Analysis Logic (The Forensic Checklist):
Brand Collision: Check for "Frankenstein" messages. For instance: does the email claim to be from Chase in the sender field but Horizon Trust Bank in the header? (High Risk Indicator).
Domain Forensics: Look for "Leetspeak" or "Descriptive Add-ons."

Examples:
Legitimate: gautierms.gov, comune.piossasco.it.
Fraud: netfl1x-renewal.com, dhl-secure-pay2087.com.

Transactional Realism: Sophisticated fraud (like the Chase example) uses "Truth-Anchors" (e.g., mentioning a Social Security deposit or a local "Gautier Pharmacy"). 
You must evaluate if these are being used to "hook" the victim into clicking a suspicious portal.

Urgency-to-Action Ratio:
Legitimate: Provides a date in the future (e.g., "next Wed," "May 12") and offers multiple ways to RSVP (link, email, or drop-in).
Fraud: Demands immediate action ("Pay now," "Update card now") to avoid a negative consequence ("avoid return," "service suspended").
Link Obfuscation: Identify when a message uses a legitimate service (like bit.ly or paypal.com) to hide the final destination or when a link looks official but points to a generic third-party domain.
Channel-Specific Logic:
If SMS: Flag short-link obfuscation, "Wrong Number" gambits, and immediate calls to action via phone numbers.
If Email: Analyze the discrepancy between the "Friendly From" name and the actual header intent (if provided), and look for professional-looking but non-standard footers/disclaimers.
Evolving Patterns: Flag "vishing" (voice-phishing) redirects or "pig-butchering" style introductory rapport building that seems out of context for a transaction.
"""

class FraudAttempt(BaseModel):
    timestamp : str = Field(
        "Timestamp of the conversation of the fraud attempt"
    )
    fraud_detail: str = Field(
        "Concise forensic reason for the flag (max 30 words)."
    )
    target_brand: Optional[str] = Field(
        description="The brand being impersonated (e.g., 'Chase', 'DHL')."
    )

class TextSummary(BaseModel):
    fraud_attempts = List[FraudAttempt]



analyzer_llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
)

structured_analyzer_llm = analyzer_llm.with_structured_output(schema=TextSummary) 


@tool
def conversation_analyzer_agent(inputs : List[str], session_id : str) -> str:
    """
    Analyzes SMS or email content to detect high-sophistication fraudulent behavior.
    
    Use this tool when you need to evaluate whether a communication is a social engineering 
    attempt, phishing, or a fraudulent transaction. This agent performs forensic linguistic 
    analysis to identify 'blended' fraud—threats that mimic legitimate business logic 
    or professional branding. 
    
    The tool identifies:
    - Subtle linguistic mimicry and typosquatting.
    - Psychological triggers (artificial urgency, scarcity).
    - Anomalous requests for sensitive data within a seemingly normal context.
    - Evolving scam patterns like vishing redirects or rapport-building (pig-butchering).
    
    Args:
        input (List[str]): The raw text of the email or SMS to be analyzed.
        session_id (str): The Langfuse session_id provided.
        
    Returns:
        str: A JSON-formatted report containing a list of FraudAttempts, which contain:
        timestamp, brief fraud description and impersonated brand. 
    """
    response = run_llm_call(session_id, structured_analyzer_llm, [SystemMessage(ANALYSIS_PROMPT),HumanMessage(f"Analyze the following conversations: {input}")])
    return response
