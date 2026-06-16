from flask import Flask, render_template, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import UnityDocsRetriever
import json
import re

app = Flask(__name__)

model = OllamaLLM(model="llama3.2", num_predict=2048, temperature=0.2)

template = """You are a Unity game engine documentation expert. Give thorough, detailed answers.

Relevant docs: {pulls}

Question: {question}

You MUST respond with ONLY a valid JSON object using exactly these three keys: "overview", "code", "directory".

Rules:
- "overview": Write 4-6 sentences explaining the concept clearly. Do not leave this short.
- "code": Write a COMPLETE, working C# Unity script with using statements, class definition, and multiple methods if needed. At least 20 lines.
- "directory": List the exact Unity docs paths like "Unity Docs > Manual > Physics" or menu paths like "Edit > Project Settings > Physics". One per line using \\n.

Do NOT include any text before or after the JSON. Do NOT use markdown fences. Output raw JSON only."""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

retriever = UnityDocsRetriever()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    print("=== /ask route hit ===")
    data = request.get_json()
    print("Data received:", data)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    pulls = retriever.search(question)
    raw = chain.invoke({"pulls": pulls, "question": question})

    print("=== RAW MODEL OUTPUT ===")
    print(raw)
    print("========================")

    def extract_field(text, key, next_key=None):
        # Match value between this key and the next key (or end)
        if next_key:
            # Grab everything between "key": " and ", "next_key"
            pattern = rf'"{key}"\s*:\s*[\"{{\[]?([\s\S]*?)[\"}}\]]?\s*,\s*"{next_key}"'
        else:
            pattern = rf'"{key}"\s*:\s*[\"{{\[]?([\s\S]*?)[\"}}\]]?\s*\}}?\s*$'
        match = re.search(pattern, text)
        if not match:
            return None
        val = match.group(1)
        # Unescape \n and \" that the model emits as literals
        val = val.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '    ')
        # Strip leading/trailing quotes the regex may have caught
        val = val.strip().strip('"')
        return val

    overview  = extract_field(raw, "overview", "code")
    code      = extract_field(raw, "code", "directory")
    directory = extract_field(raw, "directory")

    # Strip rogue leading { or [ from code
    if code:
        code = re.sub(r'^[\s\{\[]+', '', code)

    # If directory is a JSON array, extract items
    if directory and directory.strip().startswith('['):
        items = re.findall(r'"([^"]+)"', directory)
        directory = '\n'.join(items)

    result = {
        "overview":  overview  or "No overview returned.",
        "code":      code      or "// No code example returned.",
        "directory": directory or "No directory info returned."
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
