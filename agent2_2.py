"""
Vision Web Agent Prototype
General-Purpose Vision Web Agent
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
# Environment
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
4. DOM click candidates (compact list of visible interactive elements)

Decide the NEXT action.
Return JSON ONLY.

Possible actions:
- click_text
- click_candidate
- type
- scroll
- scroll_up
- go_back
- open_url
- stop

Rules:
- Prefer click_candidate with candidate_id when a good DOM candidate exists.
- If you need to move to a website, use action "open_url" with a full URL when possible.
- For click_text target, use short visible text likely on screen.
- Avoid non-visible labels and long vague phrases.
- Read Previous actions and avoid repeating failed actions.
- Use an objective-first strategy across all websites:
	1) Infer the primary objective from the task (e.g., minimum price, fastest route, highest rating, specific date).
	2) Prefer controls that directly optimize or constrain that objective.
	3) Delay secondary or cosmetic controls until the primary objective is set.
- Match controls semantically, not by exact words:
	- Use nearby synonyms, abbreviations, icon meaning, and context around the control.
	- Do not depend on one exact label string.
- Before moving on, verify the intended effect of the last important action:
	- screenshot cues (selected tab/chip, updated list/order, changed highlighted state), or
	- DOM cues (aria-selected/aria-pressed/value change).
- If the expected effect is not observed, retry with an alternative but equivalent control choice.
- If typing fails in any search/input field, vary the query format on the next try.
- Example variation order:
	1) Full text with qualifiers
	2) Core phrase without qualifiers
	3) Abbreviation/code form
	4) Short keyword form
- Do not repeat the exact same type text more than twice if it keeps failing.
- If login/sign-in page appears, prefer "go_back" or a different route.
- Add a short reflection in every action to explain intent.
- Avoid ambiguous click targets like "Dismiss" unless truly required.

Output schema (JSON only):
{
 "action": "click_text|click_candidate|type|scroll|scroll_up|go_back|open_url|stop",
 "candidate_id": "a12",
 "target": "...",
 "text": "...",
 "url": "https://...",
 "reflection": "short reason and next idea"
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
- Summarize the concrete result achieved on the webpage.
- Include key evidence visible on the final screenshot.
- If the task asks for numeric values (price/rating/count/date), report those values clearly.
- Mention uncertainty if evidence on screenshot is incomplete.

Output plain text only.
"""


TARGET_ALIASES = {
	"검색": ["Search", "Go", "Submit"],
	"확인": ["OK", "Confirm", "Apply", "Done"],
	"닫기": ["Close", "Dismiss"],
	"로그인": ["Sign in", "Log in"],
}

AMBIGUOUS_CLICK_TARGETS = {
	"dismiss",
	"close",
	"sponsored",
	"ad",
}

ACTION_TIMEOUT_MS = 8000
CLICK_ATTEMPT_TIMEOUT_MS = 2200
SHORT_WAIT_MS = 1100
POST_TYPE_WAIT_MS = 300
MAX_TARGET_FAILS = 2
MAX_SIGNATURE_FAILS = 2
MAX_HISTORY_LINES = 30
MAX_STAGNANT_SCROLLS = 2
MAX_DOM_CANDIDATES = 60
MAX_CANDIDATE_TEXT = 80
DEFAULT_START_URL = os.getenv("AGENT_START_URL", "https://www.google.com")
DEFAULT_TASK = os.getenv(
	"AGENT_TASK",
	"Find the information or action requested by the task and provide a concise final summary with evidence.",
)


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
	candidate = (text or "").strip()
	if candidate.startswith("http://") or candidate.startswith("https://"):
		return candidate
	return f"https://{candidate}"


def _find_input_box(page):
	candidates = [
		page.get_by_role("searchbox").first,
		page.get_by_role("textbox").first,
		page.get_by_role("combobox").first,
		page.locator("input[type='search']").first,
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


def _target_tokens(text):
	base = _strip_parenthetical(str(text or ""))
	parts = [p.strip() for p in re.split(r"[^\w]+", base) if p.strip()]
	return [p for p in parts if len(p) >= 2][:5]


def _find_input_box_for_target(page, target):
	tokens = _target_tokens(target)
	if target:
		candidates = [
			page.get_by_role("searchbox", name=re.compile(re.escape(str(target)), re.I)).first,
			page.get_by_role("textbox", name=re.compile(re.escape(str(target)), re.I)).first,
			page.get_by_role("combobox", name=re.compile(re.escape(str(target)), re.I)).first,
			page.get_by_placeholder(re.compile(re.escape(str(target)), re.I)).first,
			page.get_by_label(re.compile(re.escape(str(target)), re.I)).first,
		]

		for token in tokens:
			pattern = re.compile(re.escape(token), re.I)
			candidates.extend([
				page.get_by_role("searchbox", name=pattern).first,
				page.get_by_role("textbox", name=pattern).first,
				page.get_by_role("combobox", name=pattern).first,
				page.get_by_placeholder(pattern).first,
				page.get_by_label(pattern).first,
			])

		for locator in candidates:
			try:
				if locator.count() > 0:
					return locator
			except Exception:
				continue

	return _find_input_box(page)


def _strip_parenthetical(text):
	return re.sub(r"\s*\([^)]*\)", "", str(text or "")).strip()


def _build_type_variants(text):
	raw = str(text or "").strip()
	if not raw:
		return []

	base = raw.rstrip("\n").strip()
	variants = [base]

	without_paren = _strip_parenthetical(base)
	if without_paren and without_paren not in variants:
		variants.append(without_paren)

	code_match = re.search(r"\(([A-Za-z]{3,4})\)", base)
	if code_match:
		code = code_match.group(1).upper()
		if code not in variants:
			variants.append(code)

	# Short form fallback from multi-word phrase.
	words = [w for w in re.split(r"\s+", without_paren) if w]
	if len(words) >= 2:
		initials = "".join(w[0] for w in words if w[0].isalpha()).upper()
		if 2 <= len(initials) <= 4 and initials not in variants:
			variants.append(initials)

	# Keep reasonably distinct variants only.
	result = []
	for v in variants:
		v_norm = v.strip()
		if v_norm and v_norm.lower() not in {x.lower() for x in result}:
			result.append(v_norm)

	return result


def _action_signature(action):
	action_type = action.get("action", "unknown")
	token = (
		action.get("candidate_id")
		or action.get("target")
		or action.get("text")
		or action.get("url")
		or ""
	)
	return f"{action_type}:{str(token).strip().lower()}"


def _detect_page_signals(page):
	signals = []
	current_url = (page.url or "").lower()

	if any(flag in current_url for flag in ["signin", "accounts.google.com", "auth"]):
		signals.append("login_wall")

	if "captcha" in current_url:
		signals.append("captcha")

	try:
		if (
			page.get_by_text("Sign in", exact=False).count() > 0
			and page.get_by_text("password", exact=False).count() > 0
		):
			signals.append("login_wall")
	except Exception:
		pass

	return sorted(set(signals))


def _trim_history(history):
	if len(history) > MAX_HISTORY_LINES:
		return history[-MAX_HISTORY_LINES:]
	return history


def _compact_text(text, max_len=MAX_CANDIDATE_TEXT):
	t = re.sub(r"\s+", " ", str(text or "")).strip()
	if len(t) <= max_len:
		return t
	return t[: max_len - 3] + "..."


def _collect_dom_candidates(page, max_items=MAX_DOM_CANDIDATES):
	# Capture only visible interactive elements and assign temporary IDs.
	raw = page.evaluate(
		"""
		(maxItems) => {
			const selectors = [
				"button", "a[href]", "input", "select", "textarea",
				"[role='button']", "[role='link']", "[role='tab']", "[role='menuitem']",
				"[role='option']", "[role='combobox']", "[role='textbox']",
				"[onclick]", "[tabindex]"
			];

			const all = Array.from(document.querySelectorAll(selectors.join(",")));
			const isVisible = (el) => {
				const style = window.getComputedStyle(el);
				if (!style || style.visibility === "hidden" || style.display === "none") return false;
				const r = el.getBoundingClientRect();
				if (r.width < 6 || r.height < 6) return false;
				if (r.bottom < 0 || r.top > window.innerHeight) return false;
				return true;
			};

			const getText = (el) => {
				const txt = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
				if (txt) return txt;
				return (el.getAttribute("aria-label") || el.getAttribute("placeholder") || el.getAttribute("title") || "").trim();
			};

			const sorted = all
				.filter(isVisible)
				.map((el) => {
					const r = el.getBoundingClientRect();
					return { el, x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) };
				})
				.sort((a, b) => (a.y - b.y) || (a.x - b.x));

			const out = [];
			for (let i = 0; i < sorted.length; i++) {
				if (out.length >= maxItems) break;
				const item = sorted[i];
				const el = item.el;
				const id = `a${out.length + 1}`;
				el.setAttribute("data-agent-id", id);
				out.push({
					id,
					tag: (el.tagName || "").toLowerCase(),
					role: el.getAttribute("role") || "",
					text: getText(el),
					ariaLabel: el.getAttribute("aria-label") || "",
					placeholder: el.getAttribute("placeholder") || "",
					title: el.getAttribute("title") || "",
					href: el.getAttribute("href") || "",
					x: item.x,
					y: item.y,
					w: item.w,
					h: item.h
				});
			}

			return out;
		}
		""",
		max_items,
	)

	result = []
	for c in raw or []:
		result.append({
			"id": c.get("id", ""),
			"tag": c.get("tag", ""),
			"role": c.get("role", ""),
			"text": _compact_text(c.get("text", "")),
			"ariaLabel": _compact_text(c.get("ariaLabel", "")),
			"placeholder": _compact_text(c.get("placeholder", "")),
			"title": _compact_text(c.get("title", "")),
			"href": _compact_text(c.get("href", ""), 120),
			"x": c.get("x", 0),
			"y": c.get("y", 0),
		})

	return result


def _normalize_text(text):
	return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _safe_locator_value(locator):
	try:
		value = locator.input_value(timeout=300)
		return str(value or "")
	except Exception:
		pass

	try:
		value = locator.evaluate("el => (el.value ?? el.textContent ?? '').toString()")
		return str(value or "")
	except Exception:
		return ""


def _verify_type_result(typed_text, field_value_before_enter, field_value_after_action, url_before, url_after):
	typed_norm = _normalize_text(typed_text)
	before_norm = _normalize_text(field_value_before_enter)
	after_norm = _normalize_text(field_value_after_action)

	if not typed_norm:
		return True, "empty_submit"

	value_match = typed_norm in before_norm or typed_norm in after_norm
	url_changed = (url_before or "") != (url_after or "")

	if value_match:
		return True, "value_match"
	if url_changed:
		return True, "url_changed_after_type"
	return False, "no_value_match_or_navigation"


def _click_candidate(page, candidate_id):
	cid = str(candidate_id or "").strip()
	if not cid:
		return False

	try:
		locator = page.locator(f"[data-agent-id='{cid}']").first
		if locator.count() > 0:
			locator.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
			return True
	except Exception:
		return False

	return False


def _click_with_fallback(page, target):
	target = str(target or "").strip()
	if not target:
		return False, target

	if re.fullmatch(r"a\d+", target.lower()):
		if _click_candidate(page, target):
			return True, target

	targets = [target]
	without_paren = _strip_parenthetical(target)
	if without_paren and without_paren not in targets:
		targets.append(without_paren)
	targets.extend(TARGET_ALIASES.get(target, []))

	for token in _target_tokens(target):
		if token not in targets:
			targets.append(token)

	for name in targets:
		if not name:
			continue

		pattern = re.compile(re.escape(name), re.I)

		for getter in [
			lambda: page.get_by_label(pattern).first,
			lambda: page.get_by_placeholder(pattern).first,
			lambda: page.get_by_title(pattern).first,
			lambda: page.get_by_alt_text(pattern).first,
		]:
			try:
				loc = getter()
				if loc.count() > 0:
					loc.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
					return True, name
			except Exception:
				continue

		try:
			page.get_by_text(pattern).first.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
			return True, name
		except Exception:
			pass

		for role in ["button", "link", "tab", "menuitem", "option", "combobox", "textbox"]:
			try:
				page.get_by_role(role, name=pattern).first.click(timeout=CLICK_ATTEMPT_TIMEOUT_MS)
				return True, name
			except Exception:
				continue

	return False, target


def extract_json(text):
	match = re.search(r"\{.*\}", text, re.DOTALL)
	if match:
		return match.group(0)
	return None


def ask_gpt(image_path, task, history, dom_candidates):
	with open(image_path, "rb") as f:
		base64_image = base64.b64encode(f.read()).decode()

	history_text = "\n".join(history)
	dom_text = json.dumps(dom_candidates, ensure_ascii=False)

	response = client.responses.create(
		model="gpt-5-mini",
		input=[
			{"role": "system", "content": SYSTEM_PROMPT},
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

DOM click candidates (use candidate_id when possible):
{dom_text}

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

	return extract_json(response.output_text)


def ask_summary(image_path, task, history):
	with open(image_path, "rb") as f:
		base64_image = base64.b64encode(f.read()).decode()

	history_text = "\n".join(_trim_history(history))

	response = client.responses.create(
		model="gpt-5",
		input=[
			{"role": "system", "content": SUMMARY_PROMPT},
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

Based on the final screenshot and history, summarize the task result.
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
		"type_verified": None,
		"type_evidence": "",
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

	elif action_type == "click_candidate":
		candidate_id = action.get("candidate_id")
		if _click_candidate(page, candidate_id):
			print("✅ click candidate:", candidate_id)
			feedback["success"] = True
			feedback["reason"] = "clicked_candidate"
		else:
			target = action.get("target")
			if target:
				ok, clicked_target = _click_with_fallback(page, target)
				if ok:
					print("✅ click candidate fallback text:", clicked_target)
					feedback["success"] = True
					feedback["reason"] = "clicked_candidate_fallback_text"
				else:
					print("⚠️ click candidate 실패:", candidate_id)
					feedback["reason"] = "click_candidate_not_found"
			else:
				print("⚠️ click candidate 실패:", candidate_id)
				feedback["reason"] = "click_candidate_not_found"

	elif action_type == "type":
		text = (action.get("text") or "")
		target = (action.get("target") or "")
		field_value_before_enter = ""
		field_value_after_action = ""
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

				# If target is provided, focus that field first.
				if target:
					_click_with_fallback(page, target)

				if submit_after_fill and not clean_text.strip():
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

				box = _find_input_box_for_target(page, target)
				if box is None:
					raise RuntimeError("입력 가능한 검색/텍스트 박스를 찾지 못했습니다.")

				box.click(timeout=ACTION_TIMEOUT_MS)
				try:
					box.press("Control+a", timeout=ACTION_TIMEOUT_MS)
				except Exception:
					pass
				box.fill(clean_text, timeout=ACTION_TIMEOUT_MS)
				field_value_before_enter = _safe_locator_value(box)
				if submit_after_fill or clean_text:
					page.keyboard.press("Enter")
					page.wait_for_timeout(POST_TYPE_WAIT_MS)
				field_value_after_action = _safe_locator_value(box)

				print(f"⌨️ type(target={target}):", clean_text)
				feedback["success"] = True
				feedback["reason"] = "typed"
				feedback["_typed_text"] = clean_text
				feedback["_type_value_before_enter"] = field_value_before_enter
				feedback["_type_value_after_action"] = field_value_after_action

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

	if action_type == "type" and feedback.get("success"):
		type_verified, type_evidence = _verify_type_result(
			feedback.get("_typed_text", ""),
			feedback.get("_type_value_before_enter", ""),
			feedback.get("_type_value_after_action", ""),
			feedback["url_before"],
			feedback["url_after"],
		)
		feedback["type_verified"] = type_verified
		feedback["type_evidence"] = type_evidence
		if not type_verified:
			feedback["reason"] = f"{feedback['reason']}_unverified"

	feedback.pop("_typed_text", None)
	feedback.pop("_type_value_before_enter", None)
	feedback.pop("_type_value_after_action", None)
	feedback["duration_ms"] = int((time.perf_counter() - started_at) * 1000)
	return False, action, feedback


def run_agent():
	total_started = time.perf_counter()
	try:
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
			page.goto(DEFAULT_START_URL, wait_until="domcontentloaded")

			task = DEFAULT_TASK

			print("===== TASK =====")
			print(task)

			history = []
			blocked_targets = set()
			target_failures = {}
			signature_failures = {}
			type_variant_index = {}
			stagnant_scrolls = 0

			for step in range(25):
				print("\n===== STEP", step, "=====")

				screenshot_path = f"screen_{step}.png"
				page.screenshot(path=screenshot_path)
				print(f"📸 screenshot 저장: {screenshot_path} | url={page.url}")

				gpt_started = time.perf_counter()
				dom_candidates = _collect_dom_candidates(page)
				candidate_map = {c.get("id", ""): c for c in dom_candidates}
				print(f"🧩 dom candidates: {len(dom_candidates)}")
				action_json = ask_gpt(screenshot_path, task, _trim_history(history), dom_candidates)
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

				if action_obj.get("action") == "click_candidate":
					cid = str(action_obj.get("candidate_id") or "")
					if cid and not action_obj.get("target"):
						cand = candidate_map.get(cid, {})
						label = cand.get("text") or cand.get("ariaLabel") or cand.get("placeholder") or ""
						action_obj["target"] = label

				if action_obj.get("action") == "click_text":
					target_lower = str(action_obj.get("target", "")).strip().lower()
					if target_lower in AMBIGUOUS_CLICK_TARGETS:
						history.append(
							f"STEP {step}: GUARD ambiguous_click_target={action_obj.get('target')} -> replace action with scroll"
						)
						action_obj = {
							"action": "scroll",
							"reflection": "Avoid ambiguous click target and continue scanning relevant options."
						}

				current_signals = _detect_page_signals(page)
				if "login_wall" in current_signals and action_obj.get("action") in {"click_text", "type"}:
					history.append(
						f"STEP {step}: GUARD login_wall_detected -> replace action with go_back"
					)
					action_obj = {
						"action": "go_back",
						"reflection": "Login wall detected. Go back and continue without sign-in."
					}

				if action_obj.get("action") == "click_text":
					target = action_obj.get("target", "")
					if target in blocked_targets:
						history.append(
							f"STEP {step}: GUARD blocked_target={target} -> replace action with scroll_up"
						)
						action_obj = {
							"action": "scroll_up",
							"reflection": "Repeated failure on this target. Move viewport and choose another option."
						}

				if action_obj.get("action") == "click_candidate":
					candidate_id = str(action_obj.get("candidate_id") or "")
					candidate_target = str(action_obj.get("target") or "")
					if candidate_id in blocked_targets or candidate_target in blocked_targets:
						history.append(
							f"STEP {step}: GUARD blocked_candidate={candidate_id} -> replace action with scroll_up"
						)
						action_obj = {
							"action": "scroll_up",
							"reflection": "Candidate repeatedly failed. Move viewport and pick another candidate."
						}

				if action_obj.get("action") == "type":
					raw_text = str(action_obj.get("text") or "")
					target = str(action_obj.get("target") or "")
					if raw_text.strip():
						key = f"{target.lower()}|{raw_text.rstrip().lower()}"
						sig = _action_signature(action_obj)
						if signature_failures.get(sig, 0) >= 1:
							variants = _build_type_variants(raw_text)
							idx = type_variant_index.get(key, 0)
							if idx + 1 < len(variants):
								next_text = variants[idx + 1]
								type_variant_index[key] = idx + 1
								action_obj["text"] = next_text + ("\n" if raw_text.endswith("\n") else "")
								action_obj["reflection"] = (
									(action_obj.get("reflection") or "").strip()
									+ f" | retry with variant text: {next_text}"
								).strip(" |")
								history.append(
									f"STEP {step}: GUARD type_variant_applied original={raw_text!r} next={action_obj['text']!r}"
								)

				stop, executed_action, feedback = execute_action(page, action_obj)

				print(
					f"⏱️ action={feedback['duration_ms']}ms success={feedback['success']} reason={feedback['reason']}"
				)

				if executed_action is not None:
					signature = _action_signature(executed_action)

					is_stagnant_click = (
						executed_action.get("action") in {"click_text", "click_candidate"}
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

					if executed_action.get("action") in {"click_text", "click_candidate"}:
						target = executed_action.get("candidate_id") or executed_action.get("target")
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
						f"result={feedback['reason']} signals={feedback['signals']} "
						f"type_verified={feedback.get('type_verified')} type_evidence={feedback.get('type_evidence', '')}"
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
					print("===== TASK SUMMARY =====")
					print(summary_text)
					print("✅ TASK 완료")
					break

				page.wait_for_timeout(SHORT_WAIT_MS)
	finally:
		total_duration_ms = int((time.perf_counter() - total_started) * 1000)
		total_duration_sec = total_duration_ms / 1000.0
		print(f"\n⏱️ TOTAL RUNTIME: {total_duration_ms}ms ({total_duration_sec:.2f}s)")


if __name__ == "__main__":
	run_agent()
