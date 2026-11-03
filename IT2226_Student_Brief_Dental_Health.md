**STUDENT BRIEF**

**Dental Health Awareness and Affordability in Singapore**

IT2226 AI Innovator and Deep Learning Project · Project Brief

| **Item**              | **Detail**                                                                       |
|-----------------------|----------------------------------------------------------------------------------|
| Course                | IT2226 AI Innovator and Deep Learning Project                                    |
| Cohort                | Year 2, School of Information Technology, Nanyang Polytechnic                    |
| Industry context      | Inspired by an NYP x Oral Link Pte Ltd collaboration on AI oral-health screening |
| Core dataset          | Kaggle Oral Image Dataset (jhirshaikh), 4,375 images, 6 classes                  |
| Required technologies | Deep learning and agentic AI                                                     |

# 1. Background: dental health in Singapore

Dental disease in Singapore is common, mostly painless in its early stages, and expensive once it is finally treated.

**It is widespread.** The National Adult Oral Health Survey (2019, reported 2024) found up to 90% of adults have some degree of gum disease, including more than three-quarters of those aged 21 to 64. Caries among Primary 1 children rose from 50.9% (2007) to 53.1% (2016). Only 56% of people aged 55 and above keep 20 or more natural teeth.

**Awareness is low.** Early gum disease rarely hurts, so bleeding gums are often dismissed as normal. Public education has focused on decay more than gum disease, and many people spend on cosmetic treatment while skipping preventive check-ups.

**Cost is a real barrier.** Most adult dental care is delivered by private clinics and paid out of pocket. Polyclinic non-emergency appointments can involve waits of up to a year. Six in ten people eligible for government dental subsidies do not use them, and CHAS Green cardholders, Permanent Residents and pass holders receive no CHAS dental subsidy at all.

## Typical treatment fees

| **Treatment**                            | **Typical fee before GST (S$)** | **Reference**                                            |
|------------------------------------------|----------------------------------|----------------------------------------------------------|
| Scaling + polishing (one cleaning visit) | 61 to 98                         | MOH benchmark, private clinic                            |
| Simple filling                           | 65 to 160                        | MOH benchmark, private clinic                            |
| Root canal, molar                        | 872 to 1,400                     | MOH benchmark, private clinic                            |
| Permanent crown                          | 750 to 1,400                     | MOH benchmark, private clinic                            |
| Implant with crown, per unit             | 4,539 to 6,600                   | MOH 2024 average fees, public hospitals (non-subsidised) |

## Subsidies and how they vary

Subsidies exist but are hard to navigate. CHAS dental subsidies depend on card tier (Pioneer, Merdeka, Blue, Orange; Green has none) and per-procedure claim limits. From Q4 2025, CHAS Orange gained basic and preventive subsidies and restorative subsidies rose sharply (for example the Pioneer Generation crown subsidy went from S$137.50 to S$625.00). From mid-2026, Flexi-MediSave allows up to S$400 a year towards root canal treatment and crowns for patients aged 60 and above. MediSave covers only listed surgical dental procedures.

| **CHAS subsidy, max per claim (S$)** | **Pioneer** | **Merdeka** | **CHAS Blue** | **CHAS Orange** | **CHAS Green** |
|---------------------------------------|-------------|-------------|---------------|-----------------|----------------|
| Scaling                               | 40.00       | 35.00       | 30.00         | 20.00           | None           |
| Polishing                             | 30.50       | 25.50       | 20.50         | 13.50           | None           |
| Topical fluoride                      | 30.50       | 25.50       | 20.50         | 13.50           | None           |

CHAS subsidies apply to Singapore Citizens only. Scaling, polishing and fluoride may each be claimed up to 2 times per calendar year. Source: MOH, Enhanced Financing Support for CHAS Dental Procedures (effective 1 Oct 2025).

# 2. Problem statement

> Singapore residents often do not recognise early dental problems, cannot interpret a dental screening result, and delay care because they fear the cost and do not understand which subsidies apply to them. There is no single, trustworthy tool that takes a patient from "something looks wrong in my mouth" to "I know what it likely is, what treatment involves, what I will pay after subsidy, and which nearby clinic to visit".
>
> **How might we use deep learning and agentic AI to help a Singapore patient screen an oral photo for common dental conditions, understand the result in plain language, estimate their out-of-pocket cost after CHAS and MediSave, and act on it by finding a suitable clinic and appropriate home-care products?**

The problem breaks into four gaps. Your solution must close all four.

1.  **Detection gap.** Patients cannot tell calculus, caries, discolouration or an ulcer from a normal mouth, so problems go unnoticed until they hurt.

2.  **Understanding gap.** A screening label or clinical report means little to a lay patient; they need the cause, progression and urgency explained in their own language.

3.  **Cost gap.** Fees vary widely by care setting, and subsidies vary by citizenship, card tier, age and procedure, so patients cannot estimate what they will actually pay.

4.  **Action gap.** Even an informed patient must still find a suitable clinic nearby and choose the right home-care aids, which today means many separate searches.

# 3. Why this problem matters

Early detection is the cheapest dental care there is: a cleaning costs S$61 to S$98, while a molar root canal costs S$872 to S$1,400 and an implant more than S$4,500.

| **Stakeholder**                    | **Current situation**                                                                   |
|------------------------------------|-----------------------------------------------------------------------------------------|
| Patient                            | Does not notice early disease; fears unknown bills; confused by subsidy rules           |
| Elderly and lower-income residents | Highest tooth loss; most entitled to CHAS and Flexi-MediSave, yet often unaware of it   |
| Clinics                            | Patients arrive late with advanced, costly disease, or not at all                       |
| Public health                      | Up to 90% of adults have gum disease; six in ten eligible people leave subsidies unused |

# 4. Dataset

The [Kaggle Oral Image Dataset](https://www.kaggle.com/datasets/jhirshaikh/oral-image-dataset) holds 4,375 images (473.6 MB) in 6 classes under a CC0 Public Domain licence, so you may use it freely.

| **Class**           | **Images** | **Share of dataset** |
|---------------------|------------|----------------------|
| Calculus            | 1,296      | 29.6%                |
| Hypodontia          | 1,251      | 28.6%                |
| Tooth Discoloration | 1,058      | 24.2%                |
| Normal              | 286        | 6.5%                 |
| Mouth Ulcer         | 265        | 6.1%                 |
| Caries              | 219        | 5.0%                 |

![Figure 1. Images per class](./media/image1.png)

*Figure 1. Images per class*

## Facts about the data to keep in mind

- The class distribution is uneven: the smallest class (Caries, 219 images) is about one-sixth the size of the largest (Calculus, 1,296 images).

- The images are not from Singapore patients.

- Gum disease, the most common condition in Singapore adults, is not a labelled class.

- Each image carries one label, even when a mouth may show more than one condition.

- Image quality, lighting and framing vary across the dataset.

## Other public information you may use

- MOH dental fee benchmarks and public-hospital average fees

- CHAS website: eligibility, card tiers, dental subsidy amounts and the clinic locator

- CPF: MediSave and Flexi-MediSave rules for dental procedures

- HPB and NDCS oral health education material

# 5. Project requirements

Your solution must:

- Use a deep learning model that you train and evaluate on the dataset above.

- Use agentic AI that takes actions for the patient, such as searching for clinics and dental products.

- Be designed for Singapore patients, including Singapore fees and subsidy rules.

- Close all four gaps in the problem statement.

Your solution must also respect these boundaries:

- It is a screening and education aid. It must not give a diagnosis or recommend medication.

- Dental emergencies (for example facial swelling, uncontrolled bleeding, trauma) must be directed to 995 or A&E.

- Any cost figure must cite its source and be presented as a non-binding estimate.

- Personal data must be handled in line with Singapore's Personal Data Protection Act (PDPA).

- be demonstrated as a working prototype that staff could realistically use;

- be evaluated with evidence: both the model and the agent;

- consider responsible AI: fairness, privacy and consent, transparency, and the risk of the system being wrong.

- *(Blank bullet in the original document.)*

# 6. Schedule and deliverables

| **Week** | **Milestone**                                              |
|----------|------------------------------------------------------------|
| 1 to 2   | Problem framing and project proposal                       |
| 3 to 8   | Deep learning model development; mid-term review in week 8 |
| 9 to 12  | Agentic AI development                                     |
| 13 to 14 | Integration, testing and user feedback                     |
| 15 to 16 | Final presentation and live demo                           |

| **Deliverable**   | **Description**                                                           |
|-------------------|---------------------------------------------------------------------------|
| Project proposal  | Problem understanding, target users, proposed approach                    |
| Model notebook    | Training and evaluation of your deep learning model, with the saved model |
| Source code       | All application code with a README explaining how to run it               |
| Working prototype | An application a patient could use end to end                             |
| Test log          | The scenarios you tested and the results                                  |
| Final report      | Problem, design, results, limitations and responsible AI discussion       |
| Presentation      | 15-minute pitch plus live demo                                            |

| **Assessment area**                        | **Weight** |
|--------------------------------------------|------------|
| Problem understanding and solution design  | 15%        |
| Deep learning model quality and evaluation | 30%        |
| Agentic AI design, tool use and accuracy   | 30%        |
| Responsible AI, safety and testing         | 10%        |
| Report, presentation and demo              | 15%        |

*Final weights will be confirmed by your module team.*

# 7. Sources

- [Kaggle: Oral Image Dataset (jhirshaikh)](https://www.kaggle.com/datasets/jhirshaikh/oral-image-dataset)

- [NDCS: Tackling Singapore's silent epidemic of gum disease](https://www.ndcs.com.sg/news/patient-care/tackling-singapore-s-silent-epidemic-of-gum-disease)

- [HealthXchange: Say aah! Why is Singapore neglecting dental health?](https://www.healthxchange.sg/news/say-aah-why-is-singapore-neglecting-dental-health)

- [CHAS: Strengthening support for oral health](https://www.chas.sg/News-and-Events/strengthening-support-for-oral-health)

- MOH Dental Fee Benchmarks 2025 and MOH 2024 average fees at public healthcare institutions (verified 13 Sep 2026)
