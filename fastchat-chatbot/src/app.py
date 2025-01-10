from fastapi import FastAPI
from chatbot.interface import ChatbotInterface

app = FastAPI()
chatbot_interface = ChatbotInterface()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastChat chatbot!"}

@app.post("/chat")
def chat(user_input: str):
    response = chatbot_interface.get_response(user_input)
    return {"response": response}