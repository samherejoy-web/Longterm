"""Groq API client for LLM-powered data generation and refinement."""
from __future__ import annotations

import os
import time
from typing import Optional

from groq import Groq


class GroqClient:
    """Client for Groq API with rate limiting and error handling."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.request_count = 0
        self.last_request_time = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using Groq API with rate limiting."""
        # Simple rate limiting (adjust based on your plan)
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # 10 requests per second
            time.sleep(0.1)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.request_count += 1
            self.last_request_time = time.time()
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error: {e}")
            return ""

    def refine_text(
        self,
        text: str,
        language: str,
        context: str = "general",
    ) -> str:
        """Refine template-generated text using LLM."""
        system_prompt = f"""You are an expert in {language} language and content generation.
Refine the following text to make it more natural, coherent, and culturally appropriate.
Maintain factual accuracy and avoid any bias.
Context: {context}"""

        prompt = f"""Refine this {language} text while maintaining its meaning:

{text}

Provide only the refined text without any explanation."""

        return self.generate(prompt, temperature=0.5, system_prompt=system_prompt)
