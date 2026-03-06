"""
Vision Web Agent Prototype

구조

Playwright
↓
screenshot
↓
GPT Vision
↓
JSON action
↓
Playwright 실행
↓
반복
"""

import time
import json
import base64
import os
import re

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from openai import OpenAI


# ============================================================
# 1. 환경변수 로드
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# ============================================================
# 2. Agent Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a web browsing agent.

You will receive:

1. A task
2. Previous actions
3. A screenshot of the webpage

Decide the NEXT action.

Return JSON ONLY.

Possible actions:

click_text
type
scroll
stop

Examples:

{
 "action": "type",
 "text": "USD KRW"
}

{
 "action": "click_text",
 "target": "Images"
}

{
 "action": "scroll"
}

{
 "action": "stop"
}
"""


# ============================================================
# JSON 추출
# ============================================================

def extract_json(text):

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        return match.group(0)

    return None


# ============================================================
# GPT Vision
# ============================================================

def ask_gpt(image_path, task, history):

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    history_text = "\n".join(history)

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Task:
{task}

Previous actions:
{history_text}

Look at the webpage screenshot and decide the next action.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    text = response.output_text

    json_text = extract_json(text)

    return json_text


# ============================================================
# Task 결과 추출
# ============================================================

def ask_result(image_path, task):

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Based on the screenshot, answer the following task.

Task:
{task}

Return the answer only.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    return response.output_text


# ============================================================
# Action 실행
# ============================================================

def execute_action(page, action_json):

    try:
        action = json.loads(action_json)
    except:
        print("❌ JSON 파싱 실패")
        return False, None

    print("▶ 실행 action:", action)

    action_type = action.get("action")

    if action_type == "click_text":

        target = action.get("target")

        try:
            page.get_by_text(target).first.click()
            print("✅ click:", target)
        except:
            print("⚠️ click 실패:", target)

    elif action_type == "type":

        text = action.get("text")

        try:

            # 검색창 자동 탐색
            box = page.get_by_role("textbox").first

            box.click()
            box.fill(text)

            page.keyboard.press("Enter")

            print("⌨️ type:", text)

        except Exception as e:

            print("⚠️ type 실패:", e)

    elif action_type == "scroll":

        page.mouse.wheel(0, 1000)
        print("📜 scroll")

    elif action_type == "stop":

        print("🛑 stop action received")
        return True, action

    return False, action


# ============================================================
# Agent Loop
# ============================================================

def run_agent():

    with sync_playwright() as p:

        # ----------------------------------------------------
        # USER PROFILE 사용 (Google detection 감소)
        # ----------------------------------------------------
        context = p.chromium.launch_persistent_context(
            user_data_dir="chrome_profile",
            headless=False,
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled"
            ],
            viewport=None
        )

        page = context.new_page()

        # webdriver 탐지 우회
        page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        })
        """)

        page.goto("https://www.google.com")

        page.wait_for_load_state("domcontentloaded")
        #page.wait_for_timeout(3000)

        task = "Find the USD KRW exchange rate"

        print("===== TASK =====")
        print(task)

        history = []

        for step in range(10):

            print("\n===== STEP", step, "=====")

            page.screenshot(path="screen.png")

            print("📸 screenshot 저장 완료")

            action = ask_gpt("screen.png", task, history)

            print("🤖 GPT 응답:")
            print(action)

            stop, executed_action = execute_action(page, action)

            if executed_action is not None:
                history.append(json.dumps(executed_action))

            if stop:

                print("📊 Task 결과 추출 중...")

                page.screenshot(path="result.png")

                result = ask_result("result.png", task)

                print("\n===== TASK RESULT =====")
                print(result)

                print("✅ TASK 완료")
                break

            page.wait_for_timeout(2000)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    run_agent()