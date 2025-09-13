import nltk # type: ignore
import re
import random

# Download necessary NLTK data (run this once)
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')


def get_response(user_input):
    """
    Analyzes the user's input and returns a predefined response.
    """
    # Define a set of patterns and corresponding responses
    patterns = [
        (re.compile(r'\b(hi|hello|hey)\b', re.IGNORECASE), ['Hello!', 'Hi there!', 'Hey! How can I help you today?']),
        (re.compile(r'\b(how are you)\b', re.IGNORECASE), ["I'm a bot, so I'm always well!", "I'm doing great, thanks for asking!"]),
        (re.compile(r'\b(what is your name)\b', re.IGNORECASE), ['I am a simple chatbot.', 'You can call me Bot.']),
        (re.compile(r'\b(bye|goodbye|see you)\b', re.IGNORECASE), ['Goodbye!', 'Talk to you later!', 'Have a great day!']),
        (re.compile(r'\b(thank you|thanks)\b', re.IGNORECASE), ['You\'re welcome!', 'No problem!', 'My pleasure.']),
        # A general fallback response for when no pattern is matched
        (re.compile(r'.*'), ["Sorry, I don't understand that.", "Can you please rephrase that?", "That's interesting. Tell me more."])
    ]

    for pattern, responses in patterns:
        if pattern.search(user_input):
            return random.choice(responses)

def chat():
    """
    Main function to run the chatbot in a loop.
    """
    print("Bot: Hello! Type 'bye' or 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        # Check for exit commands
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("Bot: Goodbye! Have a nice day!")
            break
        
        # Get and print the bot's response
        response = get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
