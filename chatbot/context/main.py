import os
from dotenv import load_dotenv
from typing import cast, List
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool
from agents.run_context import RunContextWrapper

#Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

@cl.set_starters
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! How can I assist you today?",
            icon="🤖",
        ),
        cl.Starter(
            label="Weather",
            message="Find the weather in Karachi.",
            icon="🌤️",
        )
    ]

class MyContext:
    def __init__(self, user_id:str):
        self.user_id = user_id
        self.seen_messages = []

@function_tool
@cl.step(type="weather tool")
def get_weather(location: str, unit: str = "C") -> str:
    """
    Fetch the weather for a given location, returning a short description.
    """
    # Example logic
    return f"The weather in {location} is 22 degrees {unit}."

@function_tool
@cl.step(type="greeting tool")
def greet_user(context: RunContextWrapper[MyContext], greeting: str) -> str:
    user_id = context.context.user_id
    return f"Hello {user_id}, you said: {greeting}"

@cl.on_chat_start
async def start():
    # Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash", openai_client=external_client
    )

    config = RunConfig(
        model=model, model_provider=external_client, tracing_disabled=True
    )

    """Set up the chat sessions when a user connects."""
    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)
    agent: Agent = Agent(
        name="Assistant",
        tools=[greet_user, get_weather],
        instructions="You are a helpful assistant. Call greet_user tool to greet the user. Always greet the user when session starts.",
        model=model,
    )

    cl.user_session.set("agent", agent)

    await cl.Message(
        content="Hello! How can I assist you today?",
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messsages and generate responses."""
    #send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    #Retrieve the chat history from the user session
    chat_history = cl.user_session.get("chat_history")

    #Append the user message to the chat history
    chat_history.append({"role": "user", "content": message.content})

    my_ctx = MyContext(user_id="Hamza")

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", chat_history, "\n")

        result = await Runner.run(agent, chat_history, run_config=config, context=my_ctx)
        response = result.final_output

        #update the thinking message with the response
        msg.content = response
        await msg.update()

        chat_history.append({"role": "assistant", "content": response})

        cl.user_session.set("chat_history", chat_history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response}")
        print("History: ", chat_history)
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")

