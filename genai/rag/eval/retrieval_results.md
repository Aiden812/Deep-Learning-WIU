# Retrieval evaluation

Run 2026-11-12 · 82 entries · 306 chunks · 424 questions (each entry's own sample questions, which are not indexed).

A hit = the entry the question was written for is among the top k distinct entries returned.

| Set | Questions | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|---:|
| **All** | 424 | 60.4% | 89.4% | 95.3% |
| Dental Cost Info.txt | 118 | 56.8% | 85.6% | 93.2% |
| OralHealthEducation.txt | 177 | 62.7% | 89.3% | 94.9% |
| ProductRecommendation.txt | 59 | 54.2% | 98.3% | 100.0% |
| ReferralTreatment.txt | 43 | 53.5% | 83.7% | 95.3% |
| SupplementConditions.txt | 27 | 85.2% | 96.3% | 96.3% |

Mean reciprocal rank (top 10): 0.754

## Questions not in the top 3 (45)

| Question | Expected entry | Top result instead | Rank |
|---|---|---|---:|
| Where do I go for an implant? | Where to get dental treatment in Singapore | Dental emergencies — when to go to A&E and where | >10 |
| How much does a crown cost? | Cost — permanent crown | Extraction — what the treatment involves | 4 |
| How much is a crown at NDCS? | Cost — permanent crown | Flexi-MediSave for dental | >10 |
| Can I use Medisave for a crown? | Cost — permanent crown | Cost — dental implant (with crown) | 4 |
| What's the difference in cost between front and back tooth extraction? | Cost — simple tooth extraction | Crown — what the treatment involves | 6 |
| What's the difference in cost between simple and complex fillings? | Cost — dental fillings | Medisave for dental | 4 |
| How much is a denture at a polyclinic? | Cost — complete (full) removable denture | Replacing missing teeth — implant, bridge or denture | >10 |
| Is implant cheaper at NDCS or a private clinic? | Cost — dental implant (with crown) | Where to get dental treatment in Singapore | 6 |
| How much is a root canal at NDCS? | Cost — root canal treatment | Flexi-MediSave for dental | 4 |
| How do I know if my treatment is covered by Medisave? | How to tell if a dental treatment is Medisave-claimable — the TOSP system | CHAS Blue — dental subsidies | >10 |
| Why isn't a regular filling covered by Medisave? | How to tell if a dental treatment is Medisave-claimable — the TOSP system | Cost — dental fillings | >10 |
| How much does Medisave cover for an implant? | Medisave for dental | CHAS Blue — dental subsidies | 7 |
| Can I use Medisave for wisdom tooth extraction? | Medisave for dental | Cost — simple tooth extraction | 5 |
| What's the difference between CHAS Blue and CHAS Orange? | CHAS Orange — dental subsidies | CHAS Blue — dental subsidies | 4 |
| What is Flexi-MediSave for dental? | Flexi-MediSave for dental | Medisave for dental | 4 |
| How does Flexi-MediSave work? | Flexi-MediSave for dental | How to tell if a dental treatment is Medisave-claimable — the TOSP system | 5 |
| When does Flexi-MediSave for dental start? | Flexi-MediSave for dental | Medisave for dental | 4 |
| Should I rinse after brushing? | Brushing technique — frequency, duration, brush type | Spit but don't rinse — how to maximise fluoride benefit after brushing | 4 |
| What is the best brushing technique? | Brushing technique — frequency, duration, brush type | Interdental brushes — technique, sizing, best use cases | 7 |
| How do I use a toothbrush correctly? | Modified Bass brushing method — step-by-step technique | Manual vs electric toothbrush — which is better? | 7 |
| What kind of toothbrush should I use? | Manual vs electric toothbrush — which is better? | Manual soft-bristle toothbrush | 4 |
| Why isn't brushing enough? | Why interdental cleaning is essential — overview | Desensitising toothpaste | >10 |
| Why do I need to floss if I brush twice a day? | Why interdental cleaning is essential — overview | Brushing technique — frequency, duration, brush type | >10 |
| What is interdental cleaning? | Why interdental cleaning is essential — overview | Cost — professional cleaning (scaling and polishing) | 5 |
| Should I use interdental brushes instead of floss? | Interdental brushes — technique, sizing, best use cases | Dental floss | 5 |
| Are interdental brushes better than floss? | Interdental brushes — technique, sizing, best use cases | Interdental brush | 7 |
| Is fluoride really necessary? | How fluoride works and why it matters for cavity prevention | Root canal — what the treatment involves | 5 |
| Why do dentists recommend fluoride? | How fluoride works and why it matters for cavity prevention | Fluoride mouthwash | 10 |
| What should I drink instead of soft drinks? | Practical dietary changes for better oral health | Manual soft-bristle toothbrush | 5 |
| Can I stretch my dental visits to once a year? | How often to see a dentist — overview and rationale | Electric toothbrush | 8 |
| How often do I need a dental cleaning? | How often to see a dentist — overview and rationale | Cost — professional cleaning (scaling and polishing) | 4 |
| I had periodontal treatment — how often should I come back? | Dental recall frequency by risk level — tiered guide | Cost — dental X-rays | 6 |
| I have a broken tooth — do I need to go now? | When to see a dentist before your next scheduled recall | Dental emergencies — when to go to A&E and where | 4 |
| Why does my dentist keep mentioning smoking? | How smoking affects gum health and teeth — overview | Lifestyle and medical factors that affect gum bleeding | 8 |
| What causes plaque on teeth? | What dental plaque is and why it matters | Bleeding gums — causes, meaning, and what to do | 5 |
| How does plaque lead to cavities? | What dental plaque is and why it matters | Filling — what the treatment involves | 5 |
| Should I use interdental brushes if I have braces? | Interdental brush | Interdental brushes — technique, sizing, best use cases | 4 |
| Why do I need a crown instead of a filling? | Crown — what the treatment involves | Filling — what the treatment involves | 4 |
| Will it hurt to get a crown? | Crown — what the treatment involves | Where to get dental treatment in Singapore | 5 |
| What's the difference between a filling and a crown? | Filling — what the treatment involves | Crown — what the treatment involves | >10 |
| What is a dental implant? | Implant — what the treatment involves | How to tell if a dental treatment is Medisave-claimable — the TOSP system | 4 |
| Should I get an implant or a partial denture? | Implant — what the treatment involves | Cost — removable partial denture | >10 |
| Is the implant procedure painful? | Implant — what the treatment involves | Root canal — what the treatment involves | 5 |
| How long does a root canal-treated tooth last? | Root canal — what the treatment involves | Filling — what the treatment involves | 4 |
| My gum hurts, what should I do? | Tooth or gum pain — what to do while waiting to see a dentist | Gum recession — what it is, what causes it, and what can be done | >10 |
