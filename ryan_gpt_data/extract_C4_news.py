from datasets import load_dataset
import os

os.makedirs('data/c4', exist_ok=True)

dataset = load_dataset('allenai/c4', 'realnewslike', split='train', streaming=True)

with open('data/c4/c4.txt', 'w') as f:
    for i, item in enumerate(dataset):
        if i >= 400000:
            break
        f.write(item['text'] + '\n<|endoftext|>\n')
        if i % 50000 == 0:
            print(f'Processed {i:,} documents')

print(f'Done! Saved {i+1:,} documents')
