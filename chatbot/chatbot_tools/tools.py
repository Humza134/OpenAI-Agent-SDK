import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool

#load the environment variables from the .env file
load_dotenv()

# Set the Gemini API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@function_tool
@cl.step(type="weather tool")
def get_weather(location: str, unit: str = "C") -> str:
  """
  Fetch the weather for a given location, returning a short description.
  """
  # Example logic
  return f"The weather in {location} is 22 degrees {unit}."

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    #set up the chat session when a user connects
    """Initialize an empty chat history in the session."""
    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)

    agent:Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model)
    agent.tools.append(get_weather)
    cl.user_session.set("agent", agent)

    await cl.Message(
        content="Welcome to the Gemini Chatbot! How can I assist you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()
    #Retrieve the chat history from the session.
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chat_history") or []
    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    try:
        print("\n Calling agent with context: \n", history)
        result = Runner.run_sync(agent, history, run_config=config)

        response_content = result.final_output

        # Update the thinking message with the actual response
        msg.content = response_content
        await msg.update()

        #Append the agent response is history
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_history", history)

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(
            content=error_message
        ).send()
        print(error_message)

