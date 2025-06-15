from .gemeni import Gemini


class Generation:
    def __init__(self, model_name):
        self.model = self.create_model(model_name)

    def create_model(self, model_name):
        if 'gemini' in model_name:
            model = Gemini(model_name)
            return model

    def __call__(self,
                 prompt,
                 with_history=True):
        response = self.model(prompt, with_history=with_history)

        return response
