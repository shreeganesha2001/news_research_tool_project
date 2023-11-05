# QA Tool: News Research Tool

QA Tool is a user-friendly news research tool designed for effortless information retrieval. Users can input news article URLs and ask questions to receive relevant insights.

![](rockybot.jpg)

## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's (GooglePalm) by inputting queries and receiving answers along with source URLs.

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/codebasics/langchain.git
```

2.Navigate to the project directory:

```bash
  cd news_research_tool_project
```

3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```

4.Set up your GooglePalm API key and replace your API

## Usage/Examples

1. Run the Streamlit app by executing:

```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
