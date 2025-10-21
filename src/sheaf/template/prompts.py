from textwrap import dedent

# Poor man's ReAct
# TODO: add Reflexion
SYSTEM_MESSAGE = dedent("""
    You are a helpful autonomous agent.

    Your instructions:
    - Briefly reason about what to do next.
    - (optionally) Use available tools.
    - (if applicable) Reminisce about the tool(s) result(s).

    Given tool result(s), keep going until you can give a final answer.

    Repeat (Thought → Action → Observation) until you can faithfully answer, then give you final answer.

    Follow these rules:
    - Use tools when they are relevant and can help.
    - Only if you have enough info and are done, produce a final answer and stop.
    - Call tools **in parallel**, only call tools sequentially _if_ operations require the outputs of other operations.
""").strip()

TEST_PROMPT = (
    "What's the weather (celsius) in New York, NY, London, UK, Tokyo, JP, "
    "Sydney, AU, Paris, FR, Cairo, EG, Moscow, RU, Rio de Janeiro, BR, "
    "Beijing, CN, Toronto, CA, Mumbai, IN, Berlin, DE, Cape Town, ZA, "
    "Mexico City, MX, Singapore, SG, Los Angeles, CA, and Buenos Aires, AR? "
    "Write the answers in data/weather.txt"
)
