import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(api_key=open_api_key)

def call_openai(prompt: str, model_name: str = "gpt-3.5-turbo") -> dict:
    import time
    from openai import OpenAI
    client = OpenAI()  # make sure you set OPENAI_API_KEY in your env

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a skilled copywriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        message = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return {
            "text": message,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return {
            "text": "",
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }



def call_gemini(prompt: str, model_name: str = "gemini-pro", api_key: str = gemini_api_key) -> dict:
    """
    Call Google Gemini API and return generated text and usage metrics.

    Args:
        prompt (str): The user prompt.
        model_name (str): Gemini model to use (default: "gemini-pro").
        api_key (str): Google API Key.

    Returns:
        dict: {
            "text": generated description (str),
            "latency_ms": latency in milliseconds (float),
            "input_tokens": tokens sent (int),
            "output_tokens": tokens received (int)
        }
    """

    if not api_key:
        raise ValueError("api_key must be provided for Google Gemini API")

    genai.configure(api_key=api_key)

    # Initialize model
    model = genai.GenerativeModel(model_name)

    # Measure start time
    start_time = time.time()

    try:
        response = model.generate_content(prompt)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        text = response.text.strip() if hasattr(response, "text") else ""

        # Token accounting (Gemini SDK does not expose usage directly yet)
        input_tokens = len(prompt.split())
        output_tokens = len(text.split())

        return {
            "text": text,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "text": "",
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }
