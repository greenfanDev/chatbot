import streamlit as st
from engine import run_chat_inference

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# 1. Initialize persistent conversation state for the user's browser session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar control to reset conversation history
with st.sidebar:
    st.header("Session Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 2. Render existing conversation history on every Streamlit rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("provider"):
            st.caption(f"Served by: {msg['provider']}")

# 3. Handle new user input and build up history
if prompt := st.chat_input("Ask a question or follow-up..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_chat_inference(
                chat_history=st.session_state.messages,
                temperature=0.3,
            )

        reply_text = result["output"]
        provider_label = result["provider"]

        st.markdown(reply_text)
        st.caption(f"Served by: {provider_label}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply_text,
            "provider": provider_label,
        }
    )
