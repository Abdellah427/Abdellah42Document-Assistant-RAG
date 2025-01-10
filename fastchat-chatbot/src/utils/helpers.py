def format_response(response):
    """Format the chatbot's response for display."""
    return response.strip()

def log_interaction(user_input, bot_response):
    """Log the interaction between the user and the bot."""
    with open('interaction_log.txt', 'a') as log_file:
        log_file.write(f"User: {user_input}\nBot: {bot_response}\n\n")