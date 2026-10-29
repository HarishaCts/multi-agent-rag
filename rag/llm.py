"""LLM backend used by the Synthesis Agent (and optionally the Orchestrator).

Default `OfflineLLM` does no generation — the Synthesis Agent falls back to an
*extractive* grounded-answer path that needs no model and cannot hallucinate.
`GeminiLLM` enables abstractive generation when GOOGLE_API_KEY is set.
"""

from __future__ import annotations

import os


class OfflineLLM:
    name = "offline-extractive"
    available = False

    def generate(self, system: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("OfflineLLM cannot generate; use the extractive path.")


class GeminiLLM:  # pragma: no cover - optional / needs network + key
    available = True

    def __init__(self, model: str = "gemini-2.0-flash", timeout: int = 60):
        import google.generativeai as genai

        # REST transport + timeout: the default gRPC transport hangs behind many
        # corporate proxies; REST surfaces TLS/network errors instead of stalling.
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"], transport="rest")
        self._model = genai.GenerativeModel(model)
        self.name = model
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        resp = self._model.generate_content(
            f"{system}\n\n{user}", request_options={"timeout": self.timeout}
        )
        return resp.text


def build_llm(backend: str, gemini_model: str = "gemini-2.0-flash"):
    if backend == "gemini" and os.getenv("GOOGLE_API_KEY"):
        try:
            llm = GeminiLLM(gemini_model)
            llm.generate("You are a connectivity probe.", "Reply with: ok")  # probe once
            return llm
        except Exception as exc:
            print(f"[llm] Gemini unavailable ({type(exc).__name__}: {str(exc)[:120]}); "
                  f"using offline extractive synthesis.")
    return OfflineLLM()
