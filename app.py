from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, START, END

# --- 1. Define Agent State & Graph ---
class AgentState(BaseModel):
    query: str
    processed_step1: str = ""
    final_output: str = ""
    logs: List[str] = []

def input_processor(state: AgentState) -> Dict[str, Any]:
    logs = state.logs + ["Node 1: Processing input query..."]
    return {"processed_step1": f"Analyzed query: '{state.query}'", "logs": logs}

def output_generator(state: AgentState) -> Dict[str, Any]:
    logs = state.logs + ["Node 2: Generating final response..."]
    result = f"Phase 1 Agent Response to: '{state.processed_step1}'"
    return {"final_output": result, "logs": logs}

builder = StateGraph(AgentState)
builder.add_node("processor", input_processor)
builder.add_node("generator", output_generator)
builder.add_edge(START, "processor")
builder.add_edge("processor", "generator")
builder.add_edge("generator", END)
agent_app = builder.compile()

# --- 2. FastAPI Setup ---
app = FastAPI(title="Phase 1 Agent Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/run-agent")
async def run_agent(request: QueryRequest):
    try:
        initial_state = AgentState(query=request.query)
        result = agent_app.invoke(initial_state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Dashboard UI ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 1 Agent Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen p-6">
    <div class="max-w-4xl mx-auto space-y-6">
        <header class="border-b border-gray-800 pb-4">
            <h1 class="text-2xl font-bold text-indigo-400">Phase 1 Agent Control Center</h1>
            <p class="text-sm text-gray-400">LangGraph State Execution Viewer</p>
        </header>

        <div class="bg-gray-800 p-4 rounded-xl border border-gray-700 space-y-3">
            <label class="block text-sm font-medium text-gray-300">Prompt / Input Query</label>
            <div class="flex gap-2">
                <input id="queryInput" type="text" placeholder="Enter prompt for Phase 1 Agent..."
                       class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:border-indigo-500 text-white">
                <button onclick="runAgent()" id="runBtn"
                        class="bg-indigo-600 hover:bg-indigo-500 font-semibold px-5 py-2 rounded-lg transition-colors">
                    Run Agent
                </button>
            </div>
        </div>

        <div class="grid grid-cols-3 gap-4 text-center">
            <div id="node-start" class="bg-gray-800 p-3 rounded-lg border border-gray-700 text-sm font-mono text-gray-400">
                1. Start / Input
            </div>
            <div id="node-processor" class="bg-gray-800 p-3 rounded-lg border border-gray-700 text-sm font-mono text-gray-400">
                2. Processor Node
            </div>
            <div id="node-generator" class="bg-gray-800 p-3 rounded-lg border border-gray-700 text-sm font-mono text-gray-400">
                3. Generator Node
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-gray-800 p-4 rounded-xl border border-gray-700">
                <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Execution Logs</h2>
                <div id="logs" class="bg-gray-900 p-3 rounded-lg text-xs font-mono text-green-400 h-48 overflow-y-auto space-y-1">
                    > System ready.
                </div>
            </div>

            <div class="bg-gray-800 p-4 rounded-xl border border-gray-700">
                <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Agent Final Output</h2>
                <div id="output" class="bg-gray-900 p-3 rounded-lg text-sm font-mono text-gray-200 h-48 overflow-y-auto">
                    Waiting for execution...
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runAgent() {
            const query = document.getElementById('queryInput').value;
            const logsDiv = document.getElementById('logs');
            const outputDiv = document.getElementById('output');
            const btn = document.getElementById('runBtn');

            if (!query) return;

            btn.disabled = true;
            btn.innerText = "Running...";
            logsDiv.innerHTML = "> Initiating agent graph execution...<br>";
            outputDiv.innerText = "Processing...";

            document.getElementById('node-processor').className = "bg-indigo-900/50 border-indigo-500 p-3 rounded-lg text-sm font-mono text-indigo-300 animate-pulse";

            try {
                const res = await fetch('/api/run-agent', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });
                const data = await res.json();

                document.getElementById('node-processor').className = "bg-green-900/40 border-green-500 p-3 rounded-lg text-sm font-mono text-green-300";
                document.getElementById('node-generator').className = "bg-green-900/40 border-green-500 p-3 rounded-lg text-sm font-mono text-green-300";

                logsDiv.innerHTML = data.logs.map(l => `> ${l}`).join('<br>');
                outputDiv.innerText = data.final_output;
            } catch (err) {
                logsDiv.innerHTML += `<br><span class="text-red-400">> Error: ${err.message}</span>`;
                outputDiv.innerText = "Execution failed.";
            } finally {
                btn.disabled = false;
                btn.innerText = "Run Agent";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTML_CONTENT
