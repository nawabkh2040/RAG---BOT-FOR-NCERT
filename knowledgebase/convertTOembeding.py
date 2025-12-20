import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
	from sentence_transformers import SentenceTransformer
except Exception:
	SentenceTransformer = None

try:
	from pypdf import PdfReader
except Exception:
	PdfReader = None

try:
	import faiss
except Exception:
	faiss = None


def read_text_files(kb_dir: str, exts=None) -> List[Dict]:
	if exts is None:
		exts = {'.txt', '.md', '.html', '.pdf'}
	items = []
	for root, _, files in os.walk(kb_dir):
		for f in files:
			suf = Path(f).suffix.lower()
			if suf in exts:
				path = os.path.join(root, f)
				text = ''
				if suf == '.pdf':
					if PdfReader is None:
						# can't read PDFs; skip
						continue
					try:
						reader = PdfReader(path)
						pages = []
						for p in reader.pages:
							try:
								pages.append(p.extract_text() or '')
							except Exception:
								pages.append('')
						text = '\n'.join(pages)
					except Exception:
						text = ''
				else:
					try:
						with open(path, 'r', encoding='utf-8') as fh:
							text = fh.read()
					except UnicodeDecodeError:
						with open(path, 'r', encoding='latin-1') as fh:
							text = fh.read()
				if text and text.strip():
					items.append({'source': path, 'text': text})
	return items


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
	words = text.split()
	if not words:
		return []
	chunks = []
	start = 0
	n = len(words)
	while start < n:
		end = min(start + chunk_size, n)
		chunk = ' '.join(words[start:end])
		chunks.append(chunk)
		if end == n:
			break
		start = end - overlap if (end - overlap) > start else end
	return chunks


def build_embeddings(kb_dir: str, out_dir: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
					 chunk_size: int = 200, overlap: int = 50, batch_size: int = 64):
	if SentenceTransformer is None:
		raise RuntimeError('sentence-transformers is not installed. Install from requirements.')
	if faiss is None:
		raise RuntimeError('faiss is not installed. Install from requirements.')

	model = SentenceTransformer(model_name)

	items = read_text_files(kb_dir)
	records = []
	for it in items:
		chunks = chunk_text(it['text'], chunk_size=chunk_size, overlap=overlap)
		for i, c in enumerate(chunks):
			records.append({'source': it['source'], 'chunk_id': i, 'text': c})

	texts = [r['text'] for r in records]
	if len(texts) == 0:
		raise RuntimeError(f'No text chunks found under {kb_dir}. Check your knowledgebase files and extensions.')
	embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
	embeddings = np.array(embeddings).astype('float32')
	if embeddings.ndim == 1:
		embeddings = embeddings.reshape(1, -1)

	os.makedirs(out_dir, exist_ok=True)
	index_path = os.path.join(out_dir, 'faiss.index')
	meta_path = os.path.join(out_dir, 'metadata.json')

	dim = embeddings.shape[1]
	index = faiss.IndexFlatIP(dim)
	faiss.normalize_L2(embeddings)
	index.add(embeddings)
	faiss.write_index(index, index_path)

	# Ensure records and embeddings align; if embeddings had shape (1,dim) this will still work
	for i, r in enumerate(records):
		emb = embeddings[i]
		r['embedding_norm'] = float(np.linalg.norm(emb))

	with open(meta_path, 'w', encoding='utf-8') as fh:
		json.dump(records, fh, ensure_ascii=False, indent=2)

	return {'index_path': index_path, 'meta_path': meta_path, 'count': len(records)}


def load_index(out_dir: str):
	if faiss is None:
		raise RuntimeError('faiss is not installed. Install from requirements.')
	index_path = os.path.join(out_dir, 'faiss.index')
	meta_path = os.path.join(out_dir, 'metadata.json')
	if not os.path.exists(index_path) or not os.path.exists(meta_path):
		raise FileNotFoundError('Index or metadata not found in out_dir')
	index = faiss.read_index(index_path)
	with open(meta_path, 'r', encoding='utf-8') as fh:
		metadata = json.load(fh)
	return index, metadata


def query_index(question: str, out_dir: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', top_k: int = 5):
	if SentenceTransformer is None:
		raise RuntimeError('sentence-transformers is not installed. Install from requirements.')
	index, metadata = load_index(out_dir)
	model = SentenceTransformer(model_name)
	q_emb = model.encode([question], convert_to_numpy=True)
	q_emb = np.array(q_emb).astype('float32')
	faiss.normalize_L2(q_emb)
	D, I = index.search(q_emb, top_k)
	results = []
	for dist, idx in zip(D[0], I[0]):
		if idx < 0 or idx >= len(metadata):
			continue
		m = metadata[idx]
		results.append({'score': float(dist), 'source': m['source'], 'chunk_id': m['chunk_id'], 'text': m['text']})
	return results


def main():
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest='cmd')

	p_build = sub.add_parser('build')
	p_build.add_argument('--kb', default='..', help='Knowledgebase directory')
	p_build.add_argument('--out', default='./embeddings', help='Output directory for index and metadata')
	p_build.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
	p_build.add_argument('--chunk-size', type=int, default=200)
	p_build.add_argument('--overlap', type=int, default=50)

	p_query = sub.add_parser('query')
	p_query.add_argument('question', help='Question text to query')
	p_query.add_argument('--out', default='./embeddings', help='Output directory where index is stored')
	p_query.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
	p_query.add_argument('--top-k', type=int, default=5)

	args = parser.parse_args()
	if args.cmd == 'build':
		res = build_embeddings(args.kb, args.out, model_name=args.model, chunk_size=args.chunk_size, overlap=args.overlap)
		print('Built index:', res)
	elif args.cmd == 'query':
		results = query_index(args.question, args.out, model_name=args.model, top_k=args.top_k)
		print(json.dumps(results, ensure_ascii=False, indent=2))
	else:
		parser.print_help()


if __name__ == '__main__':
	main()

