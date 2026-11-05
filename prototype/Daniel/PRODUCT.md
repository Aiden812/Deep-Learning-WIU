# Product

## Register

product

## Users

Singapore residents (citizens, PRs and pass holders) who have noticed something in their mouth, or just want a check, and are unsure whether it matters. Many put off dental care because of cost and confusing subsidy rules. They are typically on a phone, at home, often in the evening, holding the phone up to a mirror or asking someone to take the photo. Elderly and lower-income residents matter most here: they have the most tooth loss and are the most likely to qualify for CHAS and Flexi-MediSave subsidies they don't use.

Their job: go from "something looks wrong" to "I know what it might be, how soon to act, what it will roughly cost me after subsidy, and where to go", in one sitting and without jargon.

## Product Purpose

A patient-facing screening and education aid that closes the four gaps in the IT2226 brief:

1. **Detection**: a trained image classifier screens an oral photo across six classes (calculus, caries, hypodontia, mouth ulcer, tooth discolouration, no visible condition).
2. **Understanding**: the result is explained in plain language, covering cause, how it progresses and how urgent it is.
3. **Cost**: an out-of-pocket estimate based on MOH fee benchmarks, CHAS tier, citizenship, age and Flexi-MediSave, with every figure sourced and marked non-binding.
4. **Action**: an agent finds nearby CHAS clinics and suitable home-care products.

Success means a patient finishes knowing their next step and feels the cost is manageable, and nobody mistakes the result for a diagnosis. The site must never diagnose or recommend medication, must route emergencies to 995 or A&E, and must handle photos in line with the PDPA (processed in memory, never stored without consent).

## Brand Personality

**Calm, candid, local.**

- Calm: lower the anxiety a dental worry and an unknown bill create. No alarmism and no red-siren urgency unless it really is an emergency.
- Candid: be honest about what the model can and can't see. Show confidence levels, say when it isn't sure, and note that gum disease isn't screened. Trust comes from that honesty, not from polish.
- Local: Singapore-specific throughout, covering CHAS card tiers, polyclinics, MOH benchmarks, S$ and local place names. It should read like a well-made public service, not an imported template.

Voice: a knowledgeable friend who works in public health. Short sentences, no jargon, and every number has a source.

## Anti-references

- Glossy private cosmetic-dentistry sites: stock smiles, whitening upsells, "Book now!" pressure.
- Generic SaaS health-tech landing pages: hero metrics, gradient blobs, identical three-card feature grids.
- Brandon's prototype in the same repo (warm cream neumorphism) and the plain slate/teal classifier demo. This prototype needs its own identity.
- Symptom-checker apps that alarm users with red warnings or present a single confident label as a diagnosis.

## Design Principles

1. **Honest about uncertainty.** Show the probability spread, abstain visibly when confidence is low, and state the limits of the dataset. Never present one label as fact.
2. **One next step at a time.** Each stage of the flow (photo, result, cost, clinic) has a single clear primary action. Don't flood the user with options.
3. **Every number has a source.** Fees and subsidies carry a citation and a "non-binding estimate" note next to the figure, not hidden in a footer.
4. **Safety outranks flow.** Emergency signs and the 995/A&E route sit above everything else whenever they apply.
5. **Private by default.** Ask for consent before upload, process in memory and keep no history unless the user opts in.

## Accessibility & Inclusion

WCAG 2.2 AA: ≥4.5:1 body contrast, full keyboard operation, visible focus, `prefers-reduced-motion` respected, minimum 24px targets (aim for 44px on primary actions), and status never conveyed by colour alone (urgency always has a text label).
