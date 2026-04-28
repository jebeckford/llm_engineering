#!/usr/bin/env python3
"""
Day 2 Challenge Solution: Web Page Summarizer using Ollama

This module summarizes web pages using a local Ollama LLM (llama3.2)
instead of OpenAI's paid API.

Usage:
    uv run week1/solution.py [URL]

If no URL is provided, it will summarize https://edwarddonner.com by default.

Requirements:
    - Ollama must be running locally (ollama serve)
    - llama3.2 model must be pulled (ollama pull llama3.2)
"""

import sys
from openai import OpenAI
from scraper import fetch_website_contents

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"  # Use "llama3.2:1b" for smaller computers

# System prompt for the summarizer
system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website):
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}
    ]


def summarize(url):
    """Fetch and summarize a website using Ollama."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=messages_for(website)
    )
    return response.choices[0].message.content


def main():
    """Main entry point."""
    # Use provided URL or default to edwarddonner.com
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://edwarddonner.com"

    print(f"Summarizing: {url}\n")
    print("-" * 50)

    try:
        summary = summarize(url)
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running (ollama serve) and llama3.2 is installed (ollama pull llama3.2)")
        sys.exit(1)


if __name__ == "__main__":
    main()
