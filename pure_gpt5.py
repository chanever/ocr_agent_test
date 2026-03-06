"""
Pure GPT-5 + Web Search

구조

Python
↓
GPT-5
↓
web_search tool
↓
USD/KRW 환율 반환
"""

import os
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

    response = client.responses.create(
        model="gpt-5",

        # web search tool 활성화
        tools=[
            {
                "type": "web_search"
            }
        ],

        input="""
            Find the current USD to KRW exchange rate.

            Return ONLY the number.

            Example:
            1334.25
            """
    )

    return response.output_text


# ============================================================
# 실행
# ============================================================

def run():

    print("GPT-5 + Web Search로 환율 검색 중...")

    result = ask_gpt()

    print("\n===== USD/KRW =====")
    print(result)


if __name__ == "__main__":
    run()