set -e

mkdir -p data

if [ ! -f hedc4-phonemes.txt ]; then
    wget -nc https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/hedc4-phonemes_v1.txt.7z
    7z x hedc4-phonemes_v1.txt.7z
    mv hedc4-phonemes_v1.txt hedc4-phonemes.txt
fi

if [ ! -f knesset_phonemes_v1.txt ]; then
    wget -nc https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
    7z x knesset_phonemes_v1.txt.7z
fi

uv run python -c "
import re, unicodedata
from tqdm import tqdm

def strip_nikud(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0590-\u05cf|]', '', text)
    return text.strip()

sources = ['hedc4-phonemes.txt', 'knesset_phonemes_v1.txt']
with open('data/raw.txt', 'w', encoding='utf-8') as fout:
    for path in sources:
        total = sum(1 for _ in open(path, encoding='utf-8'))
        with open(path, encoding='utf-8') as fin:
            for line in tqdm(fin, total=total, desc=path):
                hebrew = line.split('\t')[0]
                hebrew = strip_nikud(hebrew)
                if hebrew:
                    fout.write(hebrew + '\n')
"

echo "data/raw.txt ready: \$(wc -l < data/raw.txt) lines"
