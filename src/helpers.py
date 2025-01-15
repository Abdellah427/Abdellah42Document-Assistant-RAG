def preprocess_input(user_input: str) -> str:
    """
    Cleans and preprocesses the user's input by stripping whitespace and converting to lowercase.

    Args:
        user_input (str): The raw input provided by the user.

    Returns:
        str: The cleaned and formatted input.
    """
    # Clean the input by removing extra spaces and converting it to lowercase
    cleaned_input = user_input.strip().lower()
    return cleaned_input

def format_response(response: str) -> str:
    """
    Formats the response from the Mistral AI model.

    Args:
        response (str): The raw response from the AI model.

    Returns:
        str: The formatted response.
    """
    # Format the response (in this case, just return the response as is)
    formatted_response = f"{response}"
    return formatted_response
#