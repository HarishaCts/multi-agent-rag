"""Quick connectivity + credentials check for the Gemini backend.

Run this on a network WITHOUT a TLS-intercepting proxy to confirm your
GOOGLE_API_KEY works:

    python scripts/test_gemini.py

It reads the key from the environment or the local .env, makes one embedding
call and one generation call, and prints a clear PASS/FAIL for each. No secrets
are printed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def main() -> int:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("FAIL: GOOGLE_API_KEY is not set (put it in .env or the environment).")
        return 1
    print(f"GOOGLE_API_KEY detected (length {len(key)}, value hidden).")

    try:
        import google.generativeai as genai
    except ImportError:
        print("FAIL: google-generativeai not installed. Run: pip install -r requirements.txt")
        return 1

    genai.configure(api_key=key, transport="rest")
    model = os.getenv("RAG_GEMINI_MODEL", "gemini-2.0-flash")

    ok = True
    try:
        r = genai.embed_content(model="models/text-embedding-004",
                                content="hello world", request_options={"timeout": 30})
        print(f"PASS  embeddings: text-embedding-004 -> {len(r['embedding'])}-dim vector")
    except Exception as exc:
        ok = False
        print(f"FAIL  embeddings: {type(exc).__name__}: {str(exc)[:160]}")

    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content("Reply with the single word: pong",
                                   request_options={"timeout": 60})
        print(f"PASS  generation: {model} -> {resp.text.strip()[:40]!r}")
    except Exception as exc:
        ok = False
        print(f"FAIL  generation: {type(exc).__name__}: {str(exc)[:160]}")

    if ok:
        print("\nAll Gemini checks passed. The app will use Gemini automatically "
              "(RAG_*_BACKEND=gemini in your .env).")
        return 0
    print("\nSome checks failed. If you see SSLCertVerificationError, you're behind a "
          "TLS proxy — set SSL_CERT_FILE / REQUESTS_CA_BUNDLE to your corporate CA "
          "bundle, or run from an unproxied network. The app still works offline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
