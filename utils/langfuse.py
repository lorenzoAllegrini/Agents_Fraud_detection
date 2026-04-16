import os
from dotenv import load_dotenv
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler


load_dotenv()

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)


def generate_session_id():
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def invoke_langchain(model, message, langfuse_handler, session_id):
    """
    Helper function to perform a langchain invocation monitored by langfuse.
    It returns the full response object.
    """    
    return model.invoke(message, config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id},
    })



@observe()
def run_llm_call(session_id: str, model, message):
    """
    The function calls the llm, provided the session_id and the message, allowing
    to monitor the invocation with langfuse.
    It returns the full langchain response.
    """
    langfuse_handler = CallbackHandler()
    return invoke_langchain(model, message, langfuse_handler, session_id)
