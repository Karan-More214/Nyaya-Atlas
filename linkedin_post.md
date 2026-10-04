# LinkedIn post draft (edit the numbers and link)

I built NyayaAtlas ⚖️ a RAG assistant for Indian law and policy where every answer comes with its exact source and page.

Try it live: [your demo link]
Code: [your GitHub link]

The problem: legal answers from AI are only useful if you can verify them. Most chatbots give you confident text with no proof.

What I built:
🔎 Hybrid retrieval (BM25 + embeddings, fused with RRF) over the Constitution, RBI circulars and Income Tax FAQs
📄 Page-level citations for every claim
🛑 A guardrail that says "I could not find this" instead of guessing
📊 An evaluation set: Hit@5 = [X]%, MRR = [Y], abstention accuracy = [Z]%

Biggest lesson: [one honest challenge, e.g. "PDF tables extracted badly, so I tuned chunking per page" or "lowering the similarity threshold cut hallucinations but missed some valid questions"].

Stack: Python, ChromaDB, sentence-transformers, Streamlit, Claude API.

Not legal advice, just a project on making AI answers checkable.

What would you want an assistant like this to cover next?

#MachineLearning #RAG #GenAI #LLM #NLP #BuildInPublic #DataScience #India

---
Tips:
- Attach a 30 to 60 second screen recording asking one question and opening the source.
- Put the demo link in the first comment if you want better reach.
