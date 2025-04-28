import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff, RunConfig, RunContextWrapper

#Load the env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
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

    def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
        agent_name = agent.name
        print("--------------------------------")
        print(f"Handing off to {agent_name}...")
        print("--------------------------------")
        # Send a more visible message in the chat
        cl.Message(
            content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
            author="System"
        ).send()

    biling_agent = Agent(
        name="Biling Agent",
        instructions="You are a billing agent. You can help with billing inquiries.",
        model=model,
    )

    refund_agent = Agent(
        name="Refund Agent",
        instructions="You are a refund agent. You can help with refund inquiries.",
        model=model,
    )

    agent = Agent(
        name="Triage Agent",
        instructions="You are a triage agent",
        model=model,
        handoffs=[
            handoff(biling_agent, on_handoff=lambda ctx: on_handoff(biling_agent,ctx)),
            handoff(refund_agent, on_handoff=lambda ctx: on_handoff(refund_agent,ctx))
        ]
    )

    #Set session variables
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent)
    cl.user_session.set("biling_agent", biling_agent)
    cl.user_session.set("refund_agent", refund_agent)

    await cl.Message(content="Hello! How can I assist you today?").send()

@cl.on_message
async def main(msg: cl.Message):
    """Process incoming messages and generate responses."""
    #Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    #Retrieve the chat history from the user session
    chat_history = cl.user_session.get("chat_history") or []

    #Append the user message to the chat history
    chat_history.append({"role": "user", "content": msg.content})

    try:
        result = Runner.run_sync(agent, chat_history, run_config=config)
        response = result.final_output

        #update the thinking message with the actual response
        msg.content = response
        await msg.update()

        chat_history.append({"role": "assistant", "content": response})

        #update the chat history in the user session
        cl.user_session.set("chat_history", chat_history)
        print(f"History: ", chat_history)
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")



