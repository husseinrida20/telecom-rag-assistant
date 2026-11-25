# Local RAG Assistant for Telecom Operations

A fully local retrieval-augmented generation (RAG) assistant for telecom network-operations questions, plus an LLM ticket-triage classifier. Nothing leaves the machine: no API keys, no cloud calls. That matters because real support tickets often contain customer data that can't be sent to a third-party service.

## How it works

```
question ──► embed (all-MiniLM-L6-v2) ──► ChromaDB search ──► distance < 0.7 ?
                                                              │ no  → refuse (LLM never called)
                                                              │ yes → Llama 3.2 3B answers using ONLY the retrieved docs
```

- **Knowledge base:** 8 telecom-ops documents (signal quality, hardware faults, billing, outages, provisioning, performance, security).
- **Retrieval:** `sentence-transformers` embeddings indexed in ChromaDB.
- **Relevance gate:** a hard distance threshold, chosen from measured gaps. Real matches score about 0.3–0.6, off-topic queries about 1.0. Questions that don't pass are refused before generation.
- **Generation:** `llama3.2:3b` served by Ollama, prompted to answer only from the supplied context.

## Results

| Test | Result |
|---|---|
| Retrieval accuracy (10-question labeled set, incl. paraphrases and 2 out-of-scope questions) | **10/10 (100%)** |
| Generation faithfulness (spot-checked answers) | **5/6 correct** |
| Out-of-scope refusal | Refuses "What's the capital of France?" even though the LLM knows the answer, so the refusal comes from the retrieval gate, not luck |

**Known limitations, stated honestly:**
- One generation failure: for a fiber packet-loss question, the correct document was retrieved (best distance in the set), but the 3B model said there wasn't enough information. The "refuse if unsure" instruction can overcorrect on a small model.
- The ticket classifier gets the same situation wrong when it's worded differently. Output is fully deterministic at `temperature=0` (verified), so this is wording sensitivity, not randomness.
- Small knowledge base and eval set. This is a prototype, not a production benchmark.

## Repository contents

| Notebook | Purpose |
|---|---|
| `01_llm_ticket_triage.ipynb` | Few-shot prompt that classifies tickets into 5 categories with JSON output; edge cases and a determinism check |
| `02_rag_pipeline_and_evaluation.ipynb` | Knowledge base, ChromaDB index, retrieval tests, threshold-gated RAG, and evaluation |

## Run it

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```
   Keep Ollama running. It serves on `http://localhost:11434`.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in order and run all cells.

## Tech stack
Python · Ollama (Llama 3.2 3B) · ChromaDB · sentence-transformers · prompt engineering
