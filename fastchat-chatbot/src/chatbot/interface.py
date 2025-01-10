class ChatbotInterface:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def get_user_input(self):
        user_input = input("You: ")
        return user_input

    def display_response(self, response):
        print(f"Chatbot: {response}")