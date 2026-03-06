# Vision Web Agent vs Pure LLM

This project explores the difference between **Pure LLM reasoning** and **Vision-based Web Agents** when retrieving information from the internet.

The goal is to compare how different systems perform when solving real-world web tasks.

---

# Project Goal

Compare the capabilities of the following systems:

```
1. Pure LLM (LLM only)
2. Vision Web Agent (Browser + Screenshot + GPT Vision)
```

We investigate how well each system performs tasks that require:

- Web navigation
- Visual reasoning
- Interaction with websites

---

# System Overview

## 1. Pure LLM

File:

```
pure_gpt5.py
```

Architecture

```
Python
↓
GPT-5 API
↓
Text response
```

Characteristics

- No browser
- No web navigation
- No screenshots
- No real-time data access
- Only uses the model's internal knowledge

Example prompt:

```
What is the USD to KRW exchange rate?
```

Typical response:

```
I can’t access real-time data.
```

This system represents:

```
LLM knowledge only
```

---

## 2. Vision Web Agent

File:

```
agent.py
```

Architecture

```
Playwright Browser
↓
Webpage Screenshot
↓
GPT Vision
↓
JSON Action
↓
Playwright Execution
↓
Loop
```

Execution flow

```
Browser start
↓
Open webpage
↓
Screenshot
↓
Send screenshot to GPT
↓
GPT returns action
↓
Playwright executes action
↓
Repeat
```

Pseudo workflow

```
forstepinrange(N):

screenshot()

action=ask_gpt()

execute_action()

ifaction==stop:
break
```

---

# Agent Input Structure

The agent provides GPT with three pieces of information:

```
1. Task description
2. Previous actions (history)
3. Screenshot of the webpage
```

Example prompt sent to GPT:

```
Task:
Search Amazon for a product and find its reviews.

Previous actions:
STEP 1: type search query
STEP 2: click product

Look at the screenshot and decide the next action.
```

---

# Agent Action Space

The agent currently supports four actions:

```
click_text
type
scroll
stop
```

Example response from GPT:

```
{
 "action":"click_text",
 "target":"Customer reviews"
}
```

---

# Browser Automation

Browser control is implemented using **Playwright**.

Browser startup:

```
context=p.chromium.launch_persistent_context(
user_data_dir="chrome_profile",
headless=False,
args=[
"--start-maximized",
"--disable-blink-features=AutomationControlled"
    ],
viewport=None
)
```

Why persistent context?

```
- Maintain session
- Store cookies
- Reduce bot detection
```

---

# Screenshot → GPT Vision

Each step captures a screenshot of the webpage.

```
page.screenshot(path="screen.png")
```

The screenshot is encoded to base64 and sent to GPT:

```
withopen(image_path,"rb")asf:
base64_image=base64.b64encode(f.read()).decode()
```

---

# Action Execution

Playwright maps GPT actions to browser operations.

```
click_text → page.get_by_text().click()

type → search box input

scroll → mouse wheel

stop → terminate agent
```

---

# Search Box Handling

Initial implementation used fixed selectors.

Example:

```
page.fill('textarea[name="q"]',text)
```

Problem:

```
Each website uses different selectors.
```

Solution:

Automatically detect search inputs.

```
box=page.get_by_role("textbox").first
box.fill(text)
page.keyboard.press("Enter")
```

This works on most websites:

```
Google
Amazon
Bing
DuckDuckGo
```

---

# Web Search vs Vision Agent

Three main approaches exist.

| System | Capability |
| --- | --- |
| LLM only | knowledge-based |
| LLM + web_search | search engine access |
| Vision Agent | visual webpage interaction |

---

# OpenAI Web Search Tool

OpenAI provides a `web_search` tool.

Example usage:

```
response=client.responses.create(
model="gpt-5",
tools=[{"type":"web_search"}],
input="Find the current USD KRW exchange rate."
)
```

Architecture:

```
LLM
↓
Search query generation
↓
Search engine
↓
Answer synthesis
```

---

# Experiment Tasks

To evaluate the difference between systems, tasks were designed that require **visual interaction with webpages**.

Example Task:

```
Search Amazon for
"SAMSUNG 32-Inch Class Full HD F6000 Smart TV"

Open the product page.

Find customer reviews.

Summarize the reviews.
```

Expected results:

| System | Result |
| --- | --- |
| Pure LLM | ❌ Cannot access webpage |
| Vision Agent | ✅ Can navigate and read reviews |

---

# Performance Comparison

Execution time comparison:

| System | Typical Time |
| --- | --- |
| Pure LLM | 1–2 seconds |
| Vision Agent | 20–60 seconds |

Reason:

```
Browser rendering
Screenshot capture
Vision inference
Multiple reasoning steps
```

---

# Limitations

Current Vision Agent has several limitations:

### 1. Text-based clicking

```
page.get_by_text(target).click()
```

This often fails when:

```
- Text not visible
- Multiple elements share same text
```

### 2. Limited action space

Current actions:

```
click_text
type
scroll
stop
```

Missing actions:

```
go_back
open_url
click_selector
```

### 3. Bot detection

Sites like:

```
Amazon
Google
```

may trigger CAPTCHA.

---

# Future Improvements

The current agent is a **minimal research prototype**.

Future improvements include:

### 1. DOM + Vision hybrid agent

Provide both:

```
Screenshot
HTML DOM
```

to the LLM.

---

### 2. Set-of-Marks prompting

Used in research systems like:

```
VisualWebArena
WebVoyager
SeeAct
```

Example:

```
[1] Search
[2] Login
[3] Add to Cart
```

LLM selects actions via IDs.

---

### 3. Element bounding box clicking

Instead of text matching.

```
click element by visual region
```

This significantly improves reliability.

---

# Research Context

This project relates to research areas including:

```
LLM Agents
Web Automation
Multimodal Reasoning
Human-Computer Interaction
```

Similar benchmarks:

```
WebArena
VisualWebArena
WebVoyager
```

---

# Project Structure

```
vision-agent-experiment/

agent.py
pure_gpt5.py
README.md
chrome_profile/

screenshots/
```

---

# Summary

This project compares:

```
LLM knowledge
vs
LLM + Web Search
vs
Vision Web Agent
```

Key takeaway:

```
Pure LLM → cannot access real-world webpages
Vision Agent → can interact with websites visually
```

Vision-based agents open the door for **true web automation using LLMs**.