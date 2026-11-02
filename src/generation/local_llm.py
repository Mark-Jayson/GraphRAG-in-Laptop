"""Local LLM generation using mlx-lm (Apple Silicon optimized)."""
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LocalLLM:
    """Wrapper for mlx-lm local inference (Metal GPU, quantized models).

    Handles both old (temp=/top_p= kwargs) and new (sampler=) mlx-lm APIs.
    """

    def __init__(self, model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                 max_tokens: int = 512, temperature: float = 0.1,
                 top_p: float = 0.9, repetition_penalty: float = 1.1):
        from mlx_lm import load
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        logger.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        logger.info("Model loaded successfully.")

    def format_prompt(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful assistant. Answer the question based ONLY on the "
            "provided context. If the context doesn't contain the answer, say "
            "'I cannot answer based on the provided context.' Answer with just "
            "a short phrase (a few words, or yes/no) — no full sentences, no "
            "explanation."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        if hasattr(self.tokenizer, "apply_chat_template") and \
                getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return f"{system_prompt}\n\n{user_message}"

    def _generate_text(self, prompt: str) -> str:
        from mlx_lm import generate
        try:
            # Newer mlx-lm API (>=0.20): sampler + logits processors
            from mlx_lm.sample_utils import make_sampler, make_logits_processors
            sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
            processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
            return generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=self.max_tokens, sampler=sampler,
                            logits_processors=processors, verbose=False)
        except ImportError:
            # Older mlx-lm API
            return generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=self.max_tokens, temp=self.temperature,
                            top_p=self.top_p,
                            repetition_penalty=self.repetition_penalty, verbose=False)

    def generate(self, query: str, context: str) -> Dict:
        prompt = self.format_prompt(query, context)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        logger.info(f"Generating with prompt length: {prompt_tokens} tokens")

        response = self._generate_text(prompt)
        completion_tokens = len(self.tokenizer.encode(response))

        return {
            "answer": response.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
