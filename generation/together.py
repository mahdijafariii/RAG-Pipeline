import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

class TogetherModel:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        print(api_key)
        if not self.api_key:
            raise ValueError(
                "API key is required for TogetherModel. Set it via argument or TOGETHER_API_KEY environment variable.")
        self.chat_history = []
        self.client = Together(api_key=api_key) if api_key else Together()

    def __call__(self, prompt: str, stream: bool = False, with_history: bool = True):
        if with_history:
            messages = self.chat_history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        if stream:
            stream_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            return stream_response

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        response_text = response.choices[0].message.content

        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": response_text})

        return response_text

    def reset_history(self):
        self.chat_history = []
