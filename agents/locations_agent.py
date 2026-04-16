import os
from utils.langfuse import run_llm_call
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel,Field
from typing import List, Optional


class LocationSummary(BaseModel):
    primary_places : List[str] = Field(
        description = 'List of primary places visited.'
    )
    summary_of_movement : str = Field(
        description='Describe the movement patterns of the individual. Where he has been and in what periods of time. Max 100 words'
    )
    unusual_locations : List[str] = Field(
        description='List of unusual locations or weird movements conducted by the person.'
    )
    confirm_concerns : Optional[str] = Field(
        description = 'If any points of concern have been raised, address them based on the information available by your analysis.'
    )

LOCATIONS_PROMPT = """
You are a Senior Geospatial Intelligence (GEOINT) Analyst. 
Your task is to analyze a time-series of location data (Latitude, Longitude, City) to identify "Impossible Travel," account takeover indicators, 
or behavioral anomalies that suggest a user's digital identity has been compromised.

Additional User info:
{user_info}

Additional suspects and/or points of concerns (they are optional):
{concerns}

Velocity Check (Impossible Travel): Calculate the distance and time elapsed between consecutive points. If the required speed to travel between Point A and Point B exceeds commercial flight speeds (approx. 900 km/h), flag as Critical Anomaly.
Home-Base Deviation: Establish the "Centroid" (most frequent location). Flag sudden, sustained activity in a high-risk or unfamiliar city that has no historical precedent in the time-series.
Proxy/VPN Indicator: Look for "jitter" or "teleportation"—where the location jumps between distant cities and back within minutes. This suggests the use of GPS spoofing or a VPN.
Logical Consistency: Compare the movement data against the current task. (e.g., If the user is supposed to be attending a "Senior Center" in Gautier, MS, but the coordinates place them in a different state, flag it).
"""

locations_llm = ChatOpenAI(

    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
)

structured_locations_llm = locations_llm.with_structured_output(schema=LocationSummary)

@tool
def analyze_location_data(input, additional_user_info, additional_point_of_concerns, session_id):
    """
    Analyzes a user's movement timeseries (latitude, longitude, city) to detect geospatial 
    anomalies and impossible travel patterns that indicate account compromise or fraud.

    This tool should be used when you need to verify if the physical location of the user 
    aligns with their digital activity or when a suspicious communication has been flagged. 
    It performs high-velocity checks (Impossible Travel), identifies VPN/GPS spoofing, 
    and detects deviations from the user's established home-base.

    Args:
        input (str): The raw timeseries data containing timestamps, coordinates, and cities.
        additional_user_info (str): Known baseline context about the user (e.g., home city, 
                                     frequent travel hubs, or primary residence).
        additional_point_of_concerns (str): Specific suspicious events flagged by other 
                                            agents (e.g., "A login occurred from Milan at 14:00") 
                                            to be cross-referenced with the location data.
        session_id (str): Unique identifier to maintain session consistency.

    Returns:
        LocationSumamry: A structured object containing a list of common places, a summary of the movements
        a list of unusual places/movements and a confirm/dismiss of additional point of concerns (if any have been raised).
    """
    prompt = LOCATIONS_PROMPT.format(user_info = additional_user_info, concerns=additional_point_of_concerns)
    response = run_llm_call(session_id, structured_locations_llm, [SystemMessage(prompt),HumanMessage(f"Analyze the following locations info: {input}")])
    return response