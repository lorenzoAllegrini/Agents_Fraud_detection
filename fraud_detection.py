from utils.langfuse import langfuse_client, generate_session_id, run_llm_call


if __name__=='__main__':
    session_id = generate_session_id()

    # run_llm_call()

    # always remember to flush to save session
    langfuse_client.flush()
    print(session_id)