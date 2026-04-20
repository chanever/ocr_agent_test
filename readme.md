# Computer Use Agent Research Prototype

Research meeting: 2026-04-16  
Author: Jeong Haechan

This project implements a browser-based **Computer Use Agent (CUA)** prototype and compares it with a text-based LLM agent on web surfing tasks.

The main research question is:

```text
How does a text-based LLM agent differ from a vision-based LLM agent
when solving real-world web search and browsing tasks?
```

---

## Project Overview

This project compares two agent types:

```text
1. Pure LLM Agent
2. Vision-based Computer Use Agent
```

The goal is to copy, as closely as possible, how a human performs web search:

```text
Open a browser
Look at the current page
Decide what to click or type
Interact with the page
Repeat until the task is complete
```

The implemented agent uses Playwright to run a real browser. At every step, it captures a screenshot of the current webpage and sends it to GPT. GPT then reasons over the screenshot and chooses the next action, such as `click`, `type`, `scroll`, or `stop`.

---

## Research Motivation

Many LLM agents can answer questions using internal knowledge or text-based search tools. However, real web tasks often require more than text retrieval.

Real websites require:

- Visual understanding
- Page navigation
- Interaction with dynamic UI elements
- Multi-step reasoning
- Task termination after finding the needed information

This project investigates how a vision-based web agent behaves in these situations compared with a pure text-based LLM.

---

## Compared Systems

| System | Description | Web Interaction |
| --- | --- | --- |
| Pure LLM Agent | Uses only the LLM's internal knowledge | No |
| Vision CUA | Uses browser screenshots and browser actions | Yes |

---

## 1. Pure LLM Agent

File:

```text
pure_gpt5.py
```

Architecture:

```text
User task
↓
GPT-5 API
↓
Text response
```

Characteristics:

- No browser
- No webpage screenshots
- No DOM access
- No direct web navigation
- Cannot interact with websites
- Relies only on the model's internal knowledge

Example task:

```text
Find the current USD to KRW exchange rate.
```

Expected behavior:

```text
The model may explain that it cannot access real-time information.
```

This system represents:

```text
LLM reasoning without external webpage interaction
```

---

## 2. Vision-based Computer Use Agent

File:

```text
agent.py
```

Architecture:

```text
Playwright Browser
↓
Webpage Screenshot
↓
GPT Vision Reasoning
↓
JSON Action
↓
Playwright Execution
↓
Repeat
```

Execution flow:

```text
Start browser
↓
Open webpage
↓
Capture screenshot
↓
Send screenshot and task history to GPT
↓
GPT decides the next action
↓
Playwright executes the action
↓
Repeat until GPT returns stop
```

Pseudo workflow:

```python
for step in range(max_steps):
    screenshot = capture_screenshot()
    action = ask_gpt(task, history, screenshot)
    execute_action(action)

    if action["action"] == "stop":
        break
```

---

## Difference from OpenAI CUA Framework

OpenAI's CUA framework is commonly described as a general computer-using agent that can operate through visual observation and computer actions.

This project implements a browser-based Computer Use Agent with a more specific interaction method:

```text
Instead of performing tasks only through raw screen coordinates,
the agent uses Playwright to find clickable HTML elements from the webpage.
```

The agent still reasons from the current screenshot, but its actual browser actions are executed through Playwright APIs.

In other words:

```text
Screenshot → GPT reasoning → action decision → DOM-based Playwright execution
```

This makes the prototype different from a purely coordinate-based CUA. The agent sees the page visually, decides what part of the page is relevant, and then attempts to interact with clickable objects exposed by the webpage structure.

---

## Agent Input Structure

At every step, the agent sends GPT three pieces of information:

```text
1. Task description
2. Previous actions
3. Screenshot of the current webpage
```

Example prompt:

```text
Task:
Search Amazon for a product and find its customer reviews.

Previous actions:
STEP 1: typed search query
STEP 2: clicked product page

Look at the screenshot and decide the next action.
```

---

## Agent Action Space

The current agent supports the following actions:

```text
click_text
type
scroll
stop
```

Example GPT response:

```json
{
  "action": "click_text",
  "target": "Customer reviews"
}
```

Action meanings:

| Action | Meaning |
| --- | --- |
| `click_text` | Click an element containing the target text |
| `type` | Type text into an input field |
| `scroll` | Scroll the page |
| `stop` | Terminate the task |

---

## Browser Automation

Browser control is implemented using **Playwright**.

Browser startup:

```python
context = p.chromium.launch_persistent_context(
    user_data_dir="chrome_profile",
    headless=False,
    args=[
        "--start-maximized",
        "--disable-blink-features=AutomationControlled",
    ],
    viewport=None,
)
```

Why use a persistent browser context?

```text
1. Maintain browser sessions
2. Store cookies
3. Reduce repeated login or verification steps
4. Make browser behavior closer to normal human browsing
```

---

## Screenshot to GPT Vision

Each step captures a screenshot of the current webpage:

```python
page.screenshot(path="screen.png")
```

The screenshot is encoded as base64 and sent to GPT:

```python
with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()
```

GPT uses this image to understand:

- What page is currently open
- Which elements are visible
- Where the user should interact next
- Whether the task has been completed

---

## Action Execution

GPT returns a structured JSON action. Playwright maps that action to a browser operation.

```text
click_text → page.get_by_text(target).click()
type       → fill textbox and press Enter
scroll     → page.mouse.wheel()
stop       → terminate loop
```

Example:

```python
if action["action"] == "click_text":
    page.get_by_text(action["target"]).click()
```

This creates the core CUA loop:

```text
Observe → Reason → Act → Observe again
```

---

## Search Box Handling

The initial implementation used fixed selectors:

```python
page.fill('textarea[name="q"]', text)
```

Problem:

```text
Different websites use different selectors for search boxes.
```

Current approach:

```python
box = page.get_by_role("textbox").first
box.fill(text)
page.keyboard.press("Enter")
```

This works better across websites such as:

```text
Google
Amazon
Bing
DuckDuckGo
```

---

## Example Task

Example web browsing task:

```text
Search Amazon for:
"SAMSUNG 32-Inch Class Full HD F6000 Smart TV"

Open the product page.
Find customer reviews.
Summarize the reviews.
```

Expected comparison:

| System | Expected Result |
| --- | --- |
| Pure LLM Agent | Cannot directly access or inspect the current webpage |
| Vision CUA | Can open the website, navigate visually, and search for review information |

---

## Performance Comparison

Typical execution time:

| System | Typical Time |
| --- | --- |
| Pure LLM Agent | 1-2 seconds |
| Vision CUA | 20-60 seconds |

The Vision CUA is slower because it requires:

```text
Browser rendering
Screenshot capture
Vision model inference
Multiple reasoning and action steps
```

---

## Limitations

The current agent is a minimal research prototype and has several limitations.

### 1. Text-based clicking

Current implementation:

```python
page.get_by_text(target).click()
```

This can fail when:

```text
1. The target text is not visible
2. Multiple elements contain the same text
3. The visual target does not match the DOM text exactly
4. The element is hidden, delayed, or blocked by another UI layer
```

### 2. Limited action space

Current actions:

```text
click_text
type
scroll
stop
```

Missing useful actions:

```text
go_back
open_url
click_selector
click_coordinates
wait
extract_text
```

### 3. Bot detection

Some websites may trigger CAPTCHA or bot detection, especially:

```text
Amazon
Google
```

### 4. No robust task evaluation

The current prototype focuses on implementation and qualitative comparison. It does not yet include a full benchmark or automated scoring system.

---

## Future Improvements

### 1. DOM and vision hybrid agent

Provide both screenshot and DOM information to GPT:

```text
Screenshot
HTML DOM tree
Clickable element list
```

This would allow the model to reason from visual context while selecting more reliable browser actions.

### 2. Set-of-Marks prompting

Use visual labels for clickable elements, as seen in web agent research.

Example:

```text
[1] Search
[2] Login
[3] Add to Cart
```

The LLM can then select an element by ID instead of guessing text.

Related systems:

```text
VisualWebArena
WebVoyager
SeeAct
```

### 3. Element bounding box clicking

Instead of relying only on text matching, the agent could click elements by visual region or bounding box.

```text
GPT selects a visual target
↓
System maps target to element bounding box
↓
Playwright clicks the element
```

This would improve reliability for pages where the visible UI and DOM text are difficult to align.

### 4. Better task termination

The agent should more reliably decide when enough information has been found and when the task should terminate.

---

## Research Context

This project relates to:

```text
LLM Agents
Computer Use Agents
Web Automation
Multimodal Reasoning
Human-Computer Interaction
```

Related benchmarks and systems:

```text
WebArena
VisualWebArena
WebVoyager
SeeAct
OpenAI CUA
```

---

## Project Structure

```text
ocr_agent_test/
├── agent.py
├── pure_gpt5.py
├── readme.md
├── chrome_profile/
└── screenshots/
```

---

## Summary

This project implements a browser-based Computer Use Agent and compares it with a pure LLM agent.

Key takeaway:

```text
Pure LLM Agent:
Fast, but cannot directly inspect or interact with current webpages.

Vision CUA:
Slower, but can visually observe webpages and perform multi-step browser actions.
```

The implemented prototype is not a full coordinate-based desktop CUA. It is a browser-based CUA that combines screenshot-based GPT reasoning with Playwright-based DOM interaction.
