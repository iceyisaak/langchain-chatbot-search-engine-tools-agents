# Langchain Chatnbot Search Engine Tools Agents

StreamlitUI: https://langchain-chatbot-search-engine-tools-agents-hznbx9rbcsusfbmdx.streamlit.app/

Repo: https://github.com/iceyisaak/langchain-chatbot-search-engine-tools-agents


---

# LangChain Chatbot: Search Engine Tools & Agents

A robust AI chatbot built with **LangChain**, designed as an autonomous agent that can browse the live web and consult external databases to provide real-time, accurate answers.

## 🚀 Overview

This project implements a ReAct (Reasoning and Acting) agent. Unlike standard LLMs, this chatbot can recognize when it lacks information and proactively uses tools—like DuckDuckGo Search or Wikipedia—to find data before formulating a response.

## ✨ Features

* **Autonomous Agentic Reasoning:** Decides which tool to use based on the complexity of the query.
* **Built-in Tools:**
* 🔍 **DuckDuckGo Search:** For real-time web results without requiring an API key.
* 📚 **Wikipedia:** For historical facts and deep-dive summaries.
* 📄 **Arxiv:** For querying scientific papers and technical research.


* **Conversation Memory:** Maintains context across multiple exchanges.
* **Modular Architecture:** Easily switch between different LLM providers (OpenAI, Anthropic, or local models).

## 🛠️ Tech Stack

* **Framework:** [LangChain](https://github.com/langchain-ai/langchain)
* **LLMs:** OpenAI (GPT-4) or local models via Ollama
* **Search:** DuckDuckGo Search API (Community)
* **Environment:** Conda / Python

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/iceyisaak/langchain-chatbot-search-engine-tools-agents.git
cd langchain-chatbot-search-engine-tools-agents

```

### 2. Create Environment from Conda

This project includes an `environment.yml` file for easy dependency management.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate langchain-chatbot-search-engine-tools-agents

```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_actual_api_key_here
# Add other keys if using specialized tools (e.g., Arxiv)

```

## 🚀 Usage

Run the chatbot via the command line or your preferred interface:

```bash
python main.py

```

If you are using a Streamlit interface:

```bash
streamlit run app.py

```

## 🧠 How it Works

1. **Input:** User asks "Who won the game last night?"
2. **Thought:** The agent recognizes it doesn't have current sports data.
3. **Action:** It triggers the `duckduckgo_search` tool.
4. **Observation:** The tool returns the latest scores.
5. **Response:** The agent synthesizes the search results into a concise answer.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for new tool suggestions.

## 📄 License

This project is licensed under the MIT License.

---

Would you like me to help you draft the specific `environment.yml` file to match this README?
### Suggested `requirements.txt`

If you haven't created one yet, your project likely needs:

```text
python-dotenv
langchain
langchainhub
langchain_core
langchain_community
langchain_groq
langchain_openai
faiss-cpu
arxiv
wikipedia
streamlit
ddgs

```