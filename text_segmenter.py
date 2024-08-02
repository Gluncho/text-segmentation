import torch
from loader import load_model_and_tokenizer as load

DEFAULT_THRESHOLD = 0.4

class MiniSegTextSegmenter:
    def __init__(self, pretrained_dir: str = "."):
        model, tokenizer = load(pretrained_dir)
        print("Model loaded successfully")
        self.tokenizer = tokenizer
        self.model = model

    def segment_text(self, text: list[str], threshold: float | None = DEFAULT_THRESHOLD) -> list[str]:
        if threshold is None:
            threshold = DEFAULT_THRESHOLD

        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model([tokens], None)

        predictions = self._get_predictions_from_model_output(model_output, threshold)

        result = []
        l_boundary = 0
        while True:
            try:
                r_boundary = predictions.index(True, l_boundary)
            except ValueError:
                segment = " ".join(text[l_boundary:])
                result.append(segment)
                break

            segment = " ".join(text[l_boundary:r_boundary + 1])
            result.append(segment)
            l_boundary = r_boundary + 1

        return result

    def _get_predictions_from_model_output(self, model_output, threshold):
        logits = model_output.logits[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[:,-1]
        predictions = probabilities > threshold
        return predictions.tolist()

