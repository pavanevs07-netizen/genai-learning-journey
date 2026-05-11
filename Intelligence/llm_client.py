import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:

    MODEL_RATES = {
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
        "gpt-4o":      {"input": 0.002500, "output": 0.010000},
    }

    def __init__(
        self,
        model="gpt-4o-mini",
        log_path="/gdrive/My Drive/Colab Notebooks/GenAI Learning Journey/Intelligence/logs/llm_calls.jsonl"
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Check Colab Secrets or your .env file."
            )

        self.client   = OpenAI(api_key=api_key)
        self.model    = model
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def generate(
        self,
        system_prompt,
        user_message,
        temperature=0.2,
        max_tokens=500
    ):
        request_data = {
            "model":       self.model,
            "system":      system_prompt,
            "user":        user_message,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "timestamp":   datetime.now().isoformat()
        }

        start = time.time()

        raw = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        latency_ms = round((time.time() - start) * 1000)
        tokens_in  = raw.usage.prompt_tokens
        tokens_out = raw.usage.completion_tokens

        rates = self.MODEL_RATES.get(self.model, {"input": 0, "output": 0})
        cost  = round(
            (tokens_in  * rates["input"]  / 1000) +
            (tokens_out * rates["output"] / 1000),
            6
        )

        response_data = {
            "response_text": raw.choices[0].message.content,
            "tokens_used":   tokens_in + tokens_out,
            "cost_usd":      cost,
            "latency_ms":    latency_ms,
            "model_used":    self.model
        }

        self._log_call(request_data, response_data)
        return response_data

    def classify_complaint(
        self,
        complaint_text,
        vertical="banking"
    ):
        system_prompt = (
            f"You are a CX triage specialist for a {vertical} company.\n"
            "Classify the customer complaint and return ONLY valid JSON.\n"
            "No explanation. No markdown. Just the raw JSON object.\n\n"
            "Output format:\n"
            "{\n"
            '  \"category\": \"billing|account|technical|delivery|refund|other\",\n'
            '  \"urgency\": 1,\n'
            '  \"escalate_to_human\": false,\n'
            '  \"confidence\": 0.95\n'
            "}\n\n"
            "Urgency scale:\n"
            "1 = general query\n"
            "2 = minor inconvenience\n"
            "3 = moderate issue\n"
            "4 = serious complaint\n"
            "5 = legal threat or outage\n\n"
            "Few-shot examples:\n"
            'Input: \"You charged me twice and nobody is answering the phone\"\n'
            'Output: {\"category\":\"billing\",\"urgency\":4,\"escalate_to_human\":true,\"confidence\":0.95}\n\n'
            'Input: \"Where can I download my invoice?\"\n'
            'Output: {\"category\":\"account\",\"urgency\":1,\"escalate_to_human\":false,\"confidence\":0.97}\n\n'
            'Input: \"I am contacting my lawyer if this is not resolved today\"\n'
            'Output: {\"category\":\"other\",\"urgency\":5,\"escalate_to_human\":true,\"confidence\":0.99}'
        )

        result = self.generate(
            system_prompt=system_prompt,
            user_message=f"Classify this complaint: {complaint_text}",
            temperature=0.1,
            max_tokens=150
        )

        try:
            parsed = json.loads(result["response_text"])
        except json.JSONDecodeError:
            parsed = {
                "category":          "other",
                "urgency":           3,
                "escalate_to_human": True,
                "confidence":        0.0,
                "parse_error":       True,
                "raw_response":      result["response_text"]
            }

        return parsed

    def _log_call(self, request, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request":   request,
            "response":  response
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
