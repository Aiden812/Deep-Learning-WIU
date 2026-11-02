# Ethical AI checklist

- [ ] Dataset licences checked and recorded for every source in `data/raw/`
- [ ] Patient images are anonymised; no identifiable faces or metadata
- [ ] Class balance and demographic coverage documented
- [ ] Model limitations stated in the app (screening aid, not a diagnosis)
- [ ] Low-confidence and failed-quality-check results route the user to a dentist
- [ ] GenAI answers cite sourced, dated material (`genai/rag/corpus/`)
- [ ] Guardrails block medication dosing / diagnostic claims
- [ ] Cost estimates show source and retrieval date
- [ ] User images are not stored without consent
- [ ] AI use by team members logged in `docs/ai_use_logs/`
