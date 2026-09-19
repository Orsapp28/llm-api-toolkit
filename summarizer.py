"""Summarize a text file using an LLM.

Usage:
    python summarizer.py path/to/file.txt
"""

import os
import sys
import requests
from dotenv import load_dotenv
from requests.exceptions import Timeout, ConnectionError, HTTPError

load_dotenv()

LLM_API_URL = os.getenv(
    "LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")


def read_file(path: str) -> str:
    """Read the file, or exit non-zero with a clear message."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied accessing '{path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Check if empty or only whitespace
    if not content.strip():
        print(f"Error: File '{path}' is empty.", file=sys.stderr)
        sys.exit(1)

    return content


def summarize(text: str) -> tuple[str, int]:
    """Build the messages array, POST it, and return (summary_text, total_tokens)."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    system_prompt = (
        "You are an expert summarizer. Provide a concise summary in exactly 3 to 5 sentences. "
        "Do not include any introductory preamble, introductory phrases, or concluding commentary."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            LLM_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        summary = data["choices"][0]["message"]["content"].strip()
        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        return summary, total_tokens

    except Timeout:
        print(
            "Error: Request timed out. The model took too long to respond.", file=sys.stderr)
        sys.exit(1)
    except ConnectionError:
        print(
            "Error: Could not reach the LLM endpoint. Is Ollama running?", file=sys.stderr)
        sys.exit(1)
    except HTTPError as e:
        status = e.response.status_code
        if status == 401:
            print("Error: Unauthorized (401). Check your LLM_API_KEY.",
                  file=sys.stderr)
        elif status == 429:
            print(
                "Error: Rate limit reached (429). Wait a moment and retry.", file=sys.stderr)
        else:
            print(f"Error: API returned HTTP {status}.", file=sys.stderr)
        sys.exit(1)
    except (KeyError, IndexError):
        print("Error: Unexpected JSON response structure from LLM.", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <path/to/file.txt>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    text = read_file(file_path)
    summary, tokens_used = summarize(text)

    print(summary)
    print()
    print(f"Tokens used: {tokens_used}")


if __name__ == "__main__":
    main()
