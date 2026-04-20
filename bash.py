"""
Vision Web Agent Prototype
Amazon Review Task
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
# 환경변수
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# ============================================================
# Agent Prompt
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
scroll_up
go_back
open_url
stop

Rules:
- If you need to move to a website, use action "type" with a URL-like text such as "amazon.com" or "https://amazon.com".
- If you already know an exact URL, you can use action "open_url".
- For click_text target, use short visible text that likely exists on screen.
- Avoid non-visible labels and avoid long phrases.
- Read Previous actions carefully and avoid repeating failed actions.
- If login/sign-in page appears, prefer "go_back" or a different route.
- Add a short reflection in every action to explain the intent.
- Avoid ambiguous click targets like "See options" or "Dismiss" unless absolutely necessary.

Output schema (JSON only):

{
 "action": "click_text|type|scroll|scroll_up|go_back|open_url|stop",
 "target": "...",        // for click_text
 "text": "...",          // for type
 "url": "https://...",   // for open_url
 "reflection": "short reason and next idea"
}

Examples:

{
 "action": "type",
 "text": "SAMSUNG 32-Inch Class Full HD F6000 Smart TV",
 "reflection": "Found search box and entering product query"
}

{
 "action": "click_text",
 "target": "Customer reviews",
 "reflection": "Need to move from product page to review section"
}

{
 "action": "go_back",
 "reflection": "Detected sign-in wall, return to prior page"
}

{
 "action": "stop"
}
"""


SUMMARY_PROMPT = """
You are a helpful analyst.

You will receive:
1) The task
2) The agent action history
3) The final screenshot

Your job:
- Follow the task exactly.
- Report rating distribution details if visible (especially 5-star, 4-star, 3-star percentages).
- Focus on main positive and negative opinions.
- Mention uncertainty if evidence is weak.

Output plain text only.
"""


TARGET_ALIASES = {
    "Amazon 검색": ["Search Amazon", "Search"],
    "계속": ["Continue"],
    "리뷰 더보기": ["See more reviews", "See all reviews", "Customer reviews"],
}

AMBIGUOUS_CLICK_TARGETS = {
    "see options",
    "dismiss",
    "sponsored",
    "ad",
}

ACTION_TIMEOUT_MS = 5000
CLICK_ATTEMPT_TIMEOUT_MS = 1200
SHORT_WAIT_MS = 400
MAX_TARGET_FAILS = 2
MAX_SIGNATURE_FAILS = 2
MAX_HISTORY_LINES = 30
MAX_STAGNANT_SCROLLS = 2


def _looks_like_url(text):
    candidate = (text or "").strip()

    if not candidate:
        return False

    if " " in candidate:
        return False

    return (
        candidate.startswith("http://")
        or candidate.startswith("https://")
        or "." in candidate
    )


def _to_url(text):
    candidate = text.strip()

    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate

    return f"https://{candidate}"


def _find_input_box(page):
    candidates = [
        page.get_by_role("searchbox").first,
        page.get_by_role("textbox").first,
        page.get_by_role("combobox").first,
        page.locator("input[type='search']").first,
        page.locator("input[name='field-keywords']").first,
        page.locator("input[name='q']").first,
        page.locator("textarea").first,
        page.locator("input").first,
    ]

    for locator in candidates:
        try:
            if locator.count() > 0:
                return locator
        except Exception:
            continue

    return None


def _action_signature(action):
    action_type = action.get("action", "unknown")
    token = action.get("target") or action.get("text") or action.get("url") or ""
    return f"{action_type}:{str(token).strip().lower()}"


def _detect_page_signals(page):
    signals = []
    current_url = (page.url or "").lower()

    if any(flag in current_url for flag in ["/ap/signin", "signin", "auth"]):
        signals.append("login_wall")

    if "captcha" in current_url:
        signals.append("captcha")

    try:
        # Keep this strict to reduce false positives on normal Amazon pages.
        if (
            page.get_by_text("Sign in to your account", exact=False).count() > 0
            or page.get_by_text("Enter your password", exact=False).count() > 0
        ):
            signals.append("login_wall")
    except Exception:
        pass

    return sorted(set(signals))


def _trim_history(history):
    if len(history) > MAX_HISTORY_LINES:
        return history[-MAX_HISTORY_LINES:]
    return history


def _click_with_fallback(page, target):
    targets = [target]
    targets.extend(TARGET_ALIASES.get(target, []))

    for name in targets:
        if not name:
            continue

        try:
            page.get_by_text(name, exact=False).first.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
            return True, name
        except Exception:
            pass

        for role in ["button", "link", "tab", "menuitem"]:
            try:
                page.get_by_role(role, name=name, exact=False).first.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
                return True, name
            except Exception:
                continue

    return False, target


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

                        Look at the screenshot and decide the next action.
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

    return extract_json(text)


def ask_summary(image_path, task, history):

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    history_text = "\n".join(_trim_history(history))

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": SUMMARY_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Task:
{task}

Action history:
{history_text}

Based on the final screenshot and history, summarize the main customer review opinions.
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

def execute_action(page, action):

    started_at = time.perf_counter()
    before_url = page.url
    feedback = {
        "success": False,
        "reason": "unknown",
        "duration_ms": 0,
        "url_before": before_url,
        "url_after": before_url,
        "signals": _detect_page_signals(page),
        "scroll_before": None,
        "scroll_after": None,
        "scroll_delta": None,
    }

    print("▶ 실행 action:", action)

    action_type = action.get("action")

    if action_type == "click_text":

        target = action.get("target")
        ok, clicked_target = _click_with_fallback(page, target)

        if ok:
            print("✅ click:", clicked_target)
            feedback["success"] = True
            feedback["reason"] = "clicked"
        else:
            print("⚠️ click 실패:", target)
            feedback["reason"] = "click_target_not_found"

    elif action_type == "type":

        text = (action.get("text") or "")

        try:
            if _looks_like_url(text):
                url = _to_url(text)
                page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
                print("🌐 open:", url)
                feedback["success"] = True
                feedback["reason"] = "opened_url_from_type"
            else:
                submit_after_fill = text.endswith("\n")
                clean_text = text.rstrip("\n")

                # Important: when model sends only "\n", do NOT clear the current query.
                if submit_after_fill and not clean_text.strip():
                    submitted = False
                    try:
                        page.locator("input[name='field-keywords']").first.press(
                            "Enter", timeout=ACTION_TIMEOUT_MS
                        )
                        submitted = True
                    except Exception:
                        pass

                    if not submitted:
                        page.keyboard.press("Enter")

                    print("⌨️ submit: Enter-only")
                    feedback["success"] = True
                    feedback["reason"] = "submitted_enter_only"
                    
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=2000)
                    except Exception:
                        pass

                    feedback["url_after"] = page.url
                    feedback["signals"] = _detect_page_signals(page)
                    feedback["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
                    return False, action, feedback

                box = _find_input_box(page)

                if box is None:
                    raise RuntimeError("입력 가능한 검색/텍스트 박스를 찾지 못했습니다.")

                box.click(timeout=ACTION_TIMEOUT_MS)
                box.fill(clean_text, timeout=ACTION_TIMEOUT_MS)

                if submit_after_fill or clean_text:
                    page.keyboard.press("Enter")

                print("⌨️ type:", clean_text)
                feedback["success"] = True
                feedback["reason"] = "typed"

        except Exception as e:
            print("⚠️ type 실패:", e)
            feedback["reason"] = f"type_failed:{e}"

    elif action_type == "scroll":

        before_y = page.evaluate("window.scrollY")
        page.mouse.wheel(0, 1200)
        after_y = page.evaluate("window.scrollY")

        feedback["scroll_before"] = before_y
        feedback["scroll_after"] = after_y
        feedback["scroll_delta"] = after_y - before_y

        print(f"📜 scroll (delta={feedback['scroll_delta']})")
        feedback["success"] = True
        feedback["reason"] = "scrolled"

    elif action_type == "scroll_up":

        before_y = page.evaluate("window.scrollY")
        page.mouse.wheel(0, -1200)
        after_y = page.evaluate("window.scrollY")

        feedback["scroll_before"] = before_y
        feedback["scroll_after"] = after_y
        feedback["scroll_delta"] = after_y - before_y

        print(f"📜 scroll_up (delta={feedback['scroll_delta']})")
        feedback["success"] = True
        feedback["reason"] = "scrolled_up"

    elif action_type == "go_back":

        try:
            previous_url = page.url
            page.go_back(wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
            feedback["success"] = page.url != previous_url
            feedback["reason"] = "went_back" if feedback["success"] else "no_back_history"
            print("↩️ go_back")
        except Exception as e:
            print("⚠️ go_back 실패:", e)
            feedback["reason"] = f"go_back_failed:{e}"

    elif action_type == "open_url":

        try:
            url = _to_url(action.get("url", ""))
            page.goto(url, wait_until="domcontentloaded", timeout=ACTION_TIMEOUT_MS)
            print("🌐 open_url:", url)
            feedback["success"] = True
            feedback["reason"] = "opened_url"
        except Exception as e:
            print("⚠️ open_url 실패:", e)
            feedback["reason"] = f"open_url_failed:{e}"

    elif action_type == "stop":

        print("🛑 stop action received")
        feedback["success"] = True
        feedback["reason"] = "stop_received"
        feedback["url_after"] = page.url
        feedback["signals"] = _detect_page_signals(page)
        feedback["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
        return True, action, feedback

    else:
        feedback["reason"] = f"unknown_action:{action_type}"

    try:
        page.wait_for_load_state("domcontentloaded", timeout=1200)
    except Exception:
        pass

    feedback["url_after"] = page.url
    feedback["signals"] = _detect_page_signals(page)
    feedback["duration_ms"] = int((time.perf_counter() - started_at) * 1000)

    return False, action, feedback


# ============================================================
# Agent Loop
# ============================================================

def run_agent():

    with sync_playwright() as p:

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

        page.goto("https://www.amazon.com")

        page.wait_for_load_state("domcontentloaded")

        task = """
    Search Amazon for

    "SAMSUNG 32-Inch Class Full HD F6000 Smart TV"

    Open the product page.

    Find the customer rating distribution
    (5 star, 4 star, 3 star percentages).

    Summarize the customer sentiment.
"""

        print("===== TASK =====")
        print(task)

        history = []
        blocked_targets = set()
        target_failures = {}
        signature_failures = {}
        stagnant_scrolls = 0

        for step in range(20):

            print("\n===== STEP", step, "=====")

            screenshot_path = f"screen_{step}.png"

            page.screenshot(path=screenshot_path)
            print(f"📸 screenshot 저장: {screenshot_path} | url={page.url}")

            gpt_started = time.perf_counter()
            action_json = ask_gpt(screenshot_path, task, _trim_history(history))
            gpt_duration_ms = int((time.perf_counter() - gpt_started) * 1000)

            print(f"🤖 GPT 응답({gpt_duration_ms}ms):", action_json)

            if not action_json:
                history.append(f"STEP {step}: FAIL json_extract_failed")
                page.wait_for_timeout(SHORT_WAIT_MS)
                continue

            try:
                action_obj = json.loads(action_json)
            except Exception as e:
                history.append(f"STEP {step}: FAIL json_parse_failed:{e}")
                page.wait_for_timeout(SHORT_WAIT_MS)
                continue

            if "reflection" not in action_obj:
                action_obj["reflection"] = ""

            if action_obj.get("action") == "click_text":
                target_lower = str(action_obj.get("target", "")).strip().lower()
                if target_lower in AMBIGUOUS_CLICK_TARGETS:
                    history.append(
                        f"STEP {step}: GUARD ambiguous_click_target={action_obj.get('target')} -> replace action with scroll"
                    )
                    action_obj = {
                        "action": "scroll",
                        "reflection": "Avoid ambiguous click target and continue scanning product cards."
                    }

            current_signals = _detect_page_signals(page)

            # Login wall policy: fast recovery first, then avoid repeated failed route.
            if "login_wall" in current_signals and action_obj.get("action") in {"click_text", "type"}:
                history.append(
                    f"STEP {step}: GUARD login_wall_detected -> replace action with go_back"
                )
                action_obj = {
                    "action": "go_back",
                    "reflection": "Login wall detected. Go back and find another route."
                }

            if action_obj.get("action") == "click_text":
                target = action_obj.get("target", "")
                if target in blocked_targets:
                    history.append(
                        f"STEP {step}: GUARD blocked_target={target} -> replace action with go_back"
                    )
                    action_obj = {
                        "action": "go_back",
                        "reflection": "Target was repeatedly failing. Go back and choose a different entry."
                    }

            stop, executed_action, feedback = execute_action(page, action_obj)

            print(
                f"⏱️ action={feedback['duration_ms']}ms success={feedback['success']} reason={feedback['reason']}"
            )

            if executed_action is not None:

                signature = _action_signature(executed_action)

                is_stagnant_click = (
                    executed_action.get("action") == "click_text"
                    and feedback["url_before"] == feedback["url_after"]
                )

                if is_stagnant_click:
                    feedback["reason"] = f"{feedback['reason']}_stagnant"

                is_stagnant_scroll = (
                    executed_action.get("action") in {"scroll", "scroll_up"}
                    and abs(int(feedback.get("scroll_delta") or 0)) < 40
                )

                if is_stagnant_scroll:
                    stagnant_scrolls += 1
                    feedback["reason"] = f"{feedback['reason']}_stagnant"
                else:
                    if executed_action.get("action") in {"scroll", "scroll_up"}:
                        stagnant_scrolls = 0

                if stagnant_scrolls > MAX_STAGNANT_SCROLLS:
                    history.append(
                        f"STEP {step}: GUARD stagnant_scrolls={stagnant_scrolls} -> choose non-scroll action"
                    )

                if feedback["success"] and not is_stagnant_click and not is_stagnant_scroll:
                    signature_failures[signature] = 0
                else:
                    signature_failures[signature] = signature_failures.get(signature, 0) + 1

                if executed_action.get("action") == "click_text":
                    target = executed_action.get("target")
                    if target:
                        if feedback["success"] and not is_stagnant_click:
                            target_failures[target] = 0
                        else:
                            target_failures[target] = target_failures.get(target, 0) + 1
                            if target_failures[target] >= MAX_TARGET_FAILS:
                                blocked_targets.add(target)
                                print(f"🚫 반복 실패 타겟 차단: {target}")

                if signature_failures.get(signature, 0) >= MAX_SIGNATURE_FAILS:
                    history.append(
                        f"STEP {step}: GUARD repeated_signature={signature} -> choose different route"
                    )

                reflection_text = (executed_action.get("reflection") or "").strip()
                agent_feedback_reflection = (
                    f"result={feedback['reason']} signals={feedback['signals']}"
                )

                history.append(
                    "STEP {step}: action={action} success={success} url={url} reflection={reflection} agent_feedback={agent_feedback}".format(
                        step=step,
                        action=executed_action,
                        success=feedback["success"],
                        url=feedback["url_after"],
                        reflection=reflection_text,
                        agent_feedback=agent_feedback_reflection,
                    )
                )

            if stop:

                summary_image_path = "result.png"
                page.screenshot(path=summary_image_path)

                summary_started = time.perf_counter()
                summary_text = ask_summary(summary_image_path, task, history)
                summary_duration_ms = int((time.perf_counter() - summary_started) * 1000)

                print(f"\n📝 요약 생성({summary_duration_ms}ms)")
                print("===== REVIEW SUMMARY =====")
                print(summary_text)

                print("✅ TASK 완료")

                break

            page.wait_for_timeout(SHORT_WAIT_MS)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    run_agent()