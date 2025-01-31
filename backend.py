# Setup pydantic model (schema validation)
from pydantic import BaseModel
from typing import List


class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt:str
    messages: List[str]
    allow_search: bool

# Setup AI Agent from frontend request 
# create an endpoint , where my frontend will send the messages and we need fastapi
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-3.5-turbo"]
app = FastAPI(title="LangGraph AI Agent")

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request

    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error":"Invalid model name please select the valid AI model"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt  
    provider = request.model_provider
    ##Create AI agent and get response from it 
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

# Run App and explore swagger UI
if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port =8000)