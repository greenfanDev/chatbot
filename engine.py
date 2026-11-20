from openai import OpenAI, APIConnectionError, APITimeoutError
import config

#taking the current chat history and formatting that as the full context window to send to the llm inference
def _build_message_payload(
    chat_history: list[dict[str, str]],
    system_prompt: str = config.SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Prepends the system prompt and slices chat_history to prevent context overflow."""
    # Keep only standard role/content keys (strips any UI-only metadata like 'provider')
    sanitized_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chat_history
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]

    # Retain the most recent MAX_HISTORY_MESSAGES turns
    trimmed_history = sanitized_history[-10:]

    return [{"role": "system", "content": system_prompt}, *trimmed_history]


def run_chat_inference(
    chat_history: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 384,
) -> dict:
    """Runs multi-turn inference against the primary fine-tuned endpoint with Groq fallback."""
    messages = _build_message_payload(chat_history)

    primary_client = OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        timeout=25.0,
    )

    try:
        response = primary_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "output": response.choices[0].message.content.strip(),
            "provider": f"Fine-Tuned Ollama Endpoint ({config.MODEL_NAME})",
        }

    except (APIConnectionError, APITimeoutError, Exception):
        fallback_client = OpenAI(
            base_url=config.FALLBACK_BASE_URL,
            api_key=config.FALLBACK_API_KEY,
        )
        response = fallback_client.chat.completions.create(
            model=config.FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "output": response.choices[0].message.content.strip(),
            "provider": f"Groq Cloud Fallback ({config.FALLBACK_MODEL})",
        }
