"""
Pure GPT-5 + Web Search

구조

Python
↓
GPT-5
↓
web_search tool
↓
task 완료
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 환경변수 로드
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# ============================================================
# GPT + Web Search
# ============================================================

def ask_gpt():

    started_at = time.perf_counter()

    response = client.responses.create(
        model="gpt-5",

        # web search tool 활성화
        tools=[
            {
                "type": "web_search"
            }
        ],

        input="""
    Search Amazon for

    "SAMSUNG 32-Inch Class Full HD F6000 Smart TV"

    Open the product page.

    Find the customer rating distribution
    (5 star, 4 star, 3 star percentages).

    Summarize the customer sentiment.
    """
    )

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return response.output_text, elapsed_ms


# ============================================================
# 실행
# ============================================================

def run():

    total_started_at = time.perf_counter()

    print("GPT-5 + Web Search로 Amazon task 수행 중...")

    result, call_elapsed_ms = ask_gpt()

    print("\n===== PURE GPT-5 RESULT =====")
    print(result)

    total_elapsed_ms = int((time.perf_counter() - total_started_at) * 1000)

    print("\n===== RUNTIME =====")
    print(f"API call: {call_elapsed_ms} ms")
    print(f"Total: {total_elapsed_ms} ms")


if __name__ == "__main__":
    run()