def preprocess_input(user_input):
    # Function to clean and preprocess user input
    cleaned_input = user_input.strip().lower()
    return cleaned_input

def format_response(mistral_response):
    # Function to format the response from the Mistral AI model
    formatted_response = f"Chatbot: {mistral_response}"
    return formatted_response