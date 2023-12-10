from time import sleep
from typing import List, Optional, Literal, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

import threading
import uvicorn

class _RequestToolFunctionParametersProperty(BaseModel):
    title: str
    type: str

class _RequestToolFunctionParameters(BaseModel):
    type: Literal["object"]
    title: str
    required: List[str]
    properties: Dict[str, _RequestToolFunctionParametersProperty]

class _RequestToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: _RequestToolFunctionParameters

class _RequestTool(BaseModel):
    type: Literal["function"]
    function: _RequestToolFunction

class _RequestMessageFunctionCall(BaseModel):
    name: str
    arguments: str

class _RequestMessage(BaseModel):
    content: Optional[str] = None
    role: str
    name: Optional[str] = None
    function_call: Optional[_RequestMessageFunctionCall] = None

class _Request(BaseModel):
    messages: List[_RequestMessage]
    model: str
    tools: Optional[List[_RequestTool]] = None

def start_openai_api_thread(llm: Llama, host: str = "localhost", port: int = 8000):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: _Request):
        request = request.model_dump(exclude_none=True)

        # restore None content suppressed by model_dump
        messages = request["messages"]
        for message in messages:
            if not "content" in message:
                message["content"] = None

        return llm.create_chat_completion(
            messages=messages,
            tools=request["tools"] if "tools" in request else None,
        )

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port),
        daemon=True, 
    )
    
    thread.start()

    # wait for the thread to be ready
    sleep(1)
    return thread
