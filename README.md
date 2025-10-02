# ResearcherBot
An autonomous research system powered by local LLMs (via llama.cpp) and Exa API that conducts comprehensive research on any topic.

## Features
1. Multi-Agent Architecture, where one model scrapes the web, other summarizes it and the final one criticize the summary and generates the final report.
2. Local LLM inference via llama.cpp
3. Structured output by using Pydantic


### HOW TO RUN:
1. Make sure to first run the all the three llama.cpp server in the specified port i.e 8001, 8002 and 8003 using your desired models, in my case I used the qwen1.5 instruct model
2. Then just run the main.py
