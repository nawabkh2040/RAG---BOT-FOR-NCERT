import json
from typing import List, Dict
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


def load_index(out_dir: str):
    index_path = os.path.join(out_dir, 'faiss.index')
    meta_path = os.path.join(out_dir, 'metadata.json')
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError('Index or metadata not found in out_dir')
    if faiss is None:
        raise RuntimeError('faiss is not installed')
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as fh:
        metadata = json.load(fh)
    return index, metadata


def retrieve(question: str, out_dir: str = 'knowledgebase/embeddings', model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', top_k: int = 5) -> List[Dict]:
    """Return top-k retrieved chunks for `question` from the FAISS index in `out_dir`.

    Returns a list of dicts: {score, source, chunk_id, text}
    """
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers is not installed')
    index, metadata = load_index(out_dir)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([question], convert_to_numpy=True)
    q_emb = np.array(q_emb).astype('float32')
    if faiss is None:
        raise RuntimeError('faiss is not installed')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        results.append({'score': float(dist), 'source': m['source'], 'chunk_id': m['chunk_id'], 'text': m['text']})
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('question')
    parser.add_argument('--out', default='knowledgebase/embeddings')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    res = retrieve(args.question, out_dir=args.out, top_k=args.top_k)
    print(json.dumps(res, ensure_ascii=False, indent=2))
