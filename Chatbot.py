import google.generativeai as genai
import os
import textwrap
import time # For exponential backoff

# --- Configuration ---
# This function attempts to configure the Gemini API key from an environment variable.
# It's crucial for authenticating your requests to the Google Gemini service.
def configure_gemini_api():
    """Configures the Google Generative AI API key."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # If the environment variable is not set, raise an error and guide the user.
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        # Catch any exceptions during configuration and provide an informative error message.
        print(f"Error configuring Gemini API: {e}")
        print("Please ensure your GOOGLE_API_KEY environment variable is correctly set.")
        exit(1) # Exit the script as API access is essential.

# --- Model Initialization ---
# This function initializes the GenerativeModel, specifying which Gemini model to use.
# 'gemini-1.5-flash' is generally recommended for conversational applications due to its speed and cost efficiency.
def get_gemini_model(model_name='gemini-1.5-flash'):
    """Initializes and returns a GenerativeModel instance."""
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Initialized model: {model_name}")
        return model
    except Exception as e:
        # Handle errors if the model cannot be initialized (e.g., incorrect model name, access issues).
        print(f"Error initializing model '{model_name}': {e}")
        print("Please check if the model name is correct and accessible with your API key.")
        exit(1)

# --- API Call with Exponential Backoff ---
# This function implements exponential backoff, a strategy to handle transient network issues
# or API rate limits by retrying failed requests with increasing delays.
def call_api_with_backoff(chat_session, user_message, max_retries=5, initial_delay=1):
    """
    Calls the Gemini API with exponential backoff for rate limit handling.
    Args:
        chat_session: The active chat session object.
        user_message: The user's input message.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before the first retry.
    Returns:
        The API response object if successful, None otherwise.
    Raises:
        Exception: If the API call fails after all retries.
    """
    delay = initial_delay
    for i in range(max_retries):
        try:
            # Send the message to the Gemini model via the chat session.
            response = chat_session.send_message(user_message)
            return response
        except Exception as e:
            # Print a message indicating a retry attempt.
            print(f"API call failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                # If not the last retry, wait and double the delay.
                time.sleep(delay)
                delay *= 2
            else:
                # If all retries are exhausted, re-raise the exception.
                raise
    return None # This line should ideally not be reached if exceptions are re-raised on final failure.

# --- Main Chatbot Execution Logic ---
# This function orchestrates the entire chatbot interaction.
def run_chatbot():
    """Runs the interactive console chatbot."""
    configure_gemini_api() # Step 1: Configure the API key.
    model = get_gemini_model() # Step 2: Initialize the generative model.

    # Step 3: Start a new chat session. The 'history=[]' ensures a fresh conversation.
    # The 'chat' object will automatically manage and send the conversation history
    # with each subsequent 'send_message' call, enabling multi-turn dialogue.
    chat = model.start_chat(history=[])

    # Display a welcome message and instructions to the user.
    print("\n" + "="*60)
    print(" Welcome to your AI Chatbot! ".center(60))
    print(" Type 'exit' or 'quit' to end the conversation. ".center(60))
    print("="*60 + "\n")

    # Main loop for continuous conversation.
    while True:
        user_input = input("You: ").strip() # Get user input and remove leading/trailing whitespace.

        # Check for exit commands.
        if user_input.lower() in ['exit', 'quit']:
            print("\nGoodbye! Thanks for chatting.")
            break

        # Handle empty input.
        if not user_input:
            print("Bot: Please enter something to chat with me.")
            continue

        try:
            # Step 4: Send the user's message to the model and get the response.
            # The API call is wrapped with exponential backoff for robustness.
            response = call_api_with_backoff(chat, user_input)

            # Step 5: Process and print the bot's response.
            if response and response.text:
                # Use textwrap to format long responses neatly in the console.
                formatted_response = textwrap.fill(response.text, width=80)
                print("Bot:", formatted_response)
            else:
                # Inform the user if no clear response was received.
                print("Bot: I didn't get a clear response. Please try again.")

        except Exception as e:
            # Catch any unrecoverable errors during the chat process.
            print(f"Bot: An unrecoverable error occurred during the API call: {e}")
            print("The chat session might be interrupted. Please restart if needed.")
            break # Exit the loop if a critical error occurs.

    # Display a closing message.
    print("\n" + "="*60)
    print(" Chat session ended. ".center(60))
    print("="*60 + "\n")

# Ensure the chatbot runs when the script is executed directly.
if __name__ == "__main__":
    run_chatbot()

