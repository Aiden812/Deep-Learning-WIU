# Test log

| Date | Tester | Component | Test | Expected | Actual | Pass/Fail | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-12 | Brandon (with Claude) | RAG explainer: retrieval | 424 sample questions from the knowledge files (not indexed) | Right entry in top 3 | Hit@3 89.4%, Hit@5 95.3% | Pass | `genai/rag/eval/retrieval_results.md` |
| 2026-11-12 | Brandon (with Claude) | RAG explainer: safety | 39 cases: emergency, diagnosis, medicine, prompt injection, off-topic, normal | Correct route; answers cite a source | 39/39 | Pass | Development set: rules were tuned on these cases. Needs a fresh held-out set. `genai/rag/eval/safety_results.md` |
| 2026-11-12 | Brandon (with Claude) | RAG explainer: UI | Ask chip, typed cost question, emergency question in SmileCheck page | Cited answer; cost pill; red 995 card | As expected | Pass | Run in quoting mode (no LLM configured) |
