# Method Overview

1. **Problem** – Body dysmorphic disorder (BDD) and body-image distress remain undertreated because CBT providers are scarce, expensive, or stigmatized.
2. **Intervention** – A serious game presents CBT-framed scenarios, collects free-text reflections, and returns reframes/skills training via a sentiment-routing AI module.
3. **Data streams** – (a) Pre-game questionnaires (demographics + symptom scales), (b) in-game telemetry (scenario IDs, user text, AI outputs, gameplay choices), (c) post-game questionnaires on symptoms and UX.
4. **AI module** – Combines lexicon features and a dual-attention neural classifier over text/context; routes outputs to CBT feedback templates.
5. **Evaluation** – Pilot feasibility + early efficacy: usability metrics, pre/post symptom deltas, regression/ANOVA on study variables, benchmarks vs. lexicon/BERT/GPT baselines.
