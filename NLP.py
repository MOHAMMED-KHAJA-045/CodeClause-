import torch  # type: ignore
from transformers import T5ForConditionalGeneration, T5Tokenizer # type: ignore

# 1. Loading a pre-trained summarization model and its tokenizer,
# T5 is a powerful model trained on a wide range of text data.
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 2. Providing a mock legal document text.
legal_text = """
This Non-Disclosure Agreement ("Agreement") is entered into as of January 15, 2024,
by and between InnovateTech Solutions Inc., a corporation with its principal place of business at 123
Innovation Drive, Cityville, State, 12345 ("Disclosing Party"), and Sparkle Ventures LLC,
a limited liability company with its principal place of business at 678 Startup Alley,
Techburg, State, 67890 ("Receiving Party"). The parties agree to maintain strict confidentiality
regarding all proprietary information exchanged, including but not limited to business plans,
financial data, and client lists, for a period of five (5) years from the effective date of this Agreement.
The Receiving Party shall not, without the prior written consent of the Disclosing Party,
disclose or use any Confidential Information for any purpose other than evaluating a potential business
partnership. Any breach of this Agreement may result in immediate termination and legal action.
"""

# 3. Tokenizing the input text for the model
# The "summarize:" prefix is a prompt specific to the T5 model
inputs = tokenizer.encode("summarize: " + legal_text,
                          return_tensors="pt",
                          max_length=512,
                          truncation=True)

# 4. Generating the summary using the model
# 'max_length' and 'min_length' control the output size.
summary_ids = model.generate(inputs,
                             max_length=100,
                             min_length=20,
                             length_penalty=2.0,
                             num_beams=4,
                             early_stopping=True)

# 5. Decoding the summary and the results get printed in the terminal.
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Legal Document (Excerpt):\n", legal_text)
print("\n----------------------------------\n")
print("AI-Generated Summary:\n", summary_text)