def evaluate_retrieval(results, threshold=0.25):
    """
    results: list of (chunk_text, cosine_score)
    cosine_score is ~0..1 (higher is better)
    """
    if not results:
        return {"confidence": 0.0, "pass": False}

    avg = sum([score for _, score in results]) / len(results)
    return {"confidence": round(avg, 2), "pass": avg >= threshold}
