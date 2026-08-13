<!--
Copy this file to the ROOT of your Hugging Face Space repo as README.md.
The YAML front matter is how a Space is configured; it must be the first thing
in the file. Then set GROQ_API_KEY as a Space *secret* (Settings -> Variables
and secrets), never as a plain variable.

  git clone https://huggingface.co/spaces/<user>/eu-ai-act-rag hf-space
  rsync -a --exclude .git --exclude .venv --exclude data/index ./ hf-space/
  cp deploy/HF_SPACE_README.md hf-space/README.md
  cd hf-space && git add -A && git commit -m "Deploy" && git push
-->
---
title: EU AI Act RAG
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Grounded QA over the EU AI Act, with a measured eval harness
---

# EU AI Act — Retrieval-Augmented QA

Ask questions about Regulation (EU) 2024/1689. Every answer shows the passages it
was built from, cites them inline, and refuses questions the Regulation does not
cover.

The evaluation harness — retrieval quality, faithfulness, hallucination rate and
a categorised failure analysis over 45 hand-written questions — is the actual
point of the project. Numbers and methodology:
**https://github.com/aghasalim/eu-ai-act-rag**

Not legal advice.
