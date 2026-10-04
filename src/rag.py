"""Citation-first answer generation with abstention."""
import re

from src import config

SYSTEM_PROMPT = """You are NyayaAtlas, an assistant that answers questions about Indian law and policy using ONLY the numbered context passages provided.

Rules:
1. Use only facts stated in the passages. Never use outside knowledge.
2. After every claim, cite the passage number(s) in square brackets, e.g. [1] or [2][3].
3. If the passages do not contain the answer, reply exactly: "I could not find this in the indexed documents."
4. Be concise and precise. Quote article/section numbers exactly as written in the passages.
5. You provide information, not legal advice."""

NOT_FOUND = "I could not find this in the indexed documents."


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] ({r['title']}, page {r['page']})\n{r['text']}")
    return "\n\n".join(blocks)


def generate(question: str, results: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{build_context(results)}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def cited_indices(text: str, n_sources: int) -> list[int]:
    """1-based passage numbers the answer actually cites (ignores out-of-range ones)."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in found if 1 <= i <= n_sources)


def _extractive(results: list[dict], note: str) -> dict:
    return {
        "answer": note,
        "sources": results,
        "cited": [],
        "abstained": False,
        "mode": "extractive",
    }


def answer(question: str, retriever) -> dict:
    """Return {answer, sources, cited, abstained, mode}.

    `sources` is the numbered list shown to the LLM, so [n] in the answer maps to
    sources[n-1], which carries the exact file and page of that passage."""
    abstain = {"answer": NOT_FOUND, "sources": [], "cited": [], "abstained": True, "mode": "abstain"}
    if not question or not question.strip():
        return abstain

    found = retriever.search(question.strip())
    results = found["results"]

    # Guardrail 1: weak retrieval -> abstain instead of hallucinating
    if not results or found["best_similarity"] < config.MIN_SIMILARITY:
        return abstain

    # No LLM key: extractive mode (show the best passages with page numbers)
    if not config.ANTHROPIC_API_KEY:
        return _extractive(
            results,
            "No LLM API key is configured, so here are the most relevant passages "
            "from the indexed documents (see sources below).",
        )

    try:
        text = generate(question, results)
    except Exception as e:  # network / auth / rate limit: degrade, don't crash
        return _extractive(
            results,
            f"The language model is unavailable ({type(e).__name__}), so here are the "
            "most relevant passages from the indexed documents.",
        )

    # Guardrail 2: the model itself says the context is insufficient
    if not text or NOT_FOUND.lower() in text.lower():
        return abstain

    # Guardrail 3: an answer with no valid citation is ungrounded -> abstain
    cited = cited_indices(text, len(results))
    if not cited:
        return abstain

    return {"answer": text, "sources": results, "cited": cited, "abstained": False, "mode": "llm"}
