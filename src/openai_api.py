from time import sleep
from typing import List, Optional, Literal, Dict, Union
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

import threading
import uvicorn

class _ChatCompletionsRequestToolFunctionParametersProperty(BaseModel):
    title: str
    type: str

class _ChatCompletionsRequestToolFunctionParameters(BaseModel):
    type: Literal["object"]
    title: str
    required: List[str]
    properties: Dict[str, _ChatCompletionsRequestToolFunctionParametersProperty]

class _ChatCompletionsRequestToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: _ChatCompletionsRequestToolFunctionParameters

class _ChatCompletionsRequestTool(BaseModel):
    type: Literal["function"]
    function: _ChatCompletionsRequestToolFunction

class _ChatCompletionsRequestMessageFunctionCall(BaseModel):
    name: str
    arguments: str

class _ChatCompletionsRequestMessage(BaseModel):
    content: Optional[str] = None
    role: str
    name: Optional[str] = None
    function_call: Optional[_ChatCompletionsRequestMessageFunctionCall] = None

class _ChatCompletionsRequest(BaseModel):
    messages: List[_ChatCompletionsRequestMessage]
    model: str
    tools: Optional[List[_ChatCompletionsRequestTool]] = None

class _EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]    
    encoding_format: Optional[str] = None

def start_openai_api_thread(llm: Llama, host: str = "localhost", port: int = 8000):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: _ChatCompletionsRequest):
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
    
    @app.post("/v1/embeddings")
    async def embeddings(request: _EmbeddingsRequest):
        return llm.create_embedding(
            input=request.input,
            model=request.model,
        )

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port),
        daemon=True, 
    )
    
    thread.start()

    # wait for the thread to be ready
    sleep(1)
    return thread
