import hashlib
import json
from pathlib import Path
from underthesea import word_tokenize
from tqdm.auto import tqdm

def segment_all(texts, cache_dir='results/cache'):
    """Word segmentation for PhoBERT (syllables of a word joined by '_'); cached by content hash."""
    digest = hashlib.md5('\n'.join(texts).encode('utf-8')).hexdigest()[:12]
    cache_file = Path(cache_dir) / f'segmented_{digest}.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding='utf-8'))

    segmented = [word_tokenize(t, format='text') for t in tqdm(texts, desc='segment')]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(segmented, ensure_ascii=False), encoding='utf-8')
    return segmented
