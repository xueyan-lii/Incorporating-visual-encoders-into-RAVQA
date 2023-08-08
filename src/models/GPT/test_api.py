# Import the os package
import os

# Import the openai package
import openai

# From the IPython.display package, import display and Markdown
from IPython.display import display, Markdown

# Import yfinance as yf
import yfinance as yf

# Set openai.api_key to the OPENAI environment variable
openai.api_key = os.environ["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"]

response = openai.ChatCompletion.create(
              model="MODEL_NAME",
              messages=[{"role": "system", "content": 'SPECIFY HOW THE AI ASSISTANT SHOULD BEHAVE'},
                        {"role": "user", "content": 'SPECIFY WANT YOU WANT THE AI ASSISTANT TO SAY'}
              ])