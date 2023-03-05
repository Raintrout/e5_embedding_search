from embedding import EmbeddingPipeline
from time import process_time

import faiss
import numpy as np

index = faiss.IndexFlatL2(1024)

text = []
overflows = 0
with open('data/samples.jsonl', 'r') as file:
    import json
    for entry in file.read().split('\n'):
        if len(entry):
            entry = json.loads(entry)['text']
            text.append('passage: ' + entry)

            if len(entry.split(' ')) > 500:
                overflows += 1

percentage_overflow = round(100*overflows/len(text), 1)
print(f"Percentage overflow: {percentage_overflow}")

pipe = EmbeddingPipeline()

MAX_BATCH = 16
for i in range(0, len(text), MAX_BATCH):
    embeddings = pipe.embedding(text[i:i+MAX_BATCH]).detach().cpu().numpy()
    index.add(embeddings)


query = input('query: ')
while query != "Q":
    search = pipe.embedding('query: '+ query).detach().cpu().numpy()
    D, I = index.search(search, 3)

    for i in I[0]:
        print(text[i])

    query = input('query: ')