from time import sleep
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from model_config import get_config
from llama_cpp import Llama
from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import threading
import uvicorn

#config = get_config("OpenHermes-2.5-Mistral-7B")
#config = get_config("Starling-LM-7B-alpha")
#config = get_config("Mistral-7B-Instruct-v0.1")
config = get_config("OpenHermes-2.5-neural-chat-7B-v3-1-7B")

llm = Llama(**config)
app = FastAPI()

class ChatMessage(BaseModel):
    content: str
    role: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str

@app.post("/v1/chat/completions")
async def completions(request: ChatRequest):
    return llm.create_chat_completion([{ 
        "role": m.role, 
        "content": m.content
    } for m in request.messages])

if __name__ == "__main__":
    threading.Thread(
        target=lambda: uvicorn.run(app, host="localhost", port=8000),
        daemon=True, 
    ).start()

    sleep(1)

    llm_config={
        "cache_seed": 42,
        "config_list": [{
            "base_url": "http://localhost:8000/v1",
            "model": "dontcare",
            "api_key": "dontcare",
        }],
    }

    human = UserProxyAgent(
        name="Human",
        system_message="A human appreciating good poetry.",
        human_input_mode="ALWAYS",
        code_execution_config={
            "last_n_messages": 2, 
            "work_dir": "groupchat"
        },
    )

    critic = AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message="A poety critic, you assess the poem quality (structure, rimes, respect of the human request) and give feedback to the poet.",
    )

    poet = AssistantAgent(
        name="Poet",
        system_message="You're a poet, you take the human input to write poems. You take critic as your mentor and modify your poems according to his remarks.",
        llm_config=llm_config,
    )

    groupchat = GroupChat (
        agents=[human, poet, critic], 
        messages=[], 
        max_round=12,
        speaker_selection_method="manual",
    )

    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    human.initiate_chat(
        manager, 
        message="Write a poem about poultry."
    )
