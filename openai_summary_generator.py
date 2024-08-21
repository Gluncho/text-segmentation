from pydantic import BaseModel
from openai import OpenAI

DEFAULT_MODEL_NAME = "gpt-4o-2024-08-06"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that extracts main topic and summary from the paragraph I provide."


class ParagraphSummary(BaseModel):
    topic: str
    summary: str


class OpenAiSummaryGenerator:
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_summaries(
        self, paragraphs: list[str], system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> list[ParagraphSummary]:
        return [
            self._generate_summary_from_paragraph(paragraph, system_prompt)
            for paragraph in paragraphs
        ]

    def _generate_summary_from_paragraph(
        self, paragraph: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> ParagraphSummary:

        completion = self.client.beta.chat.completions.parse(
            model=DEFAULT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": paragraph},
            ],
            response_format=ParagraphSummary,
        )
        return completion.choices[0].message.parsed
