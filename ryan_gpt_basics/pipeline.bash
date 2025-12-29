#!/bin/bash
set -e
cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 1: Extract Wikipedia (50k articles)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_data/extract_wikipedia.py \
    --max_articles 50000

echo "=========================================="
echo "STEP 2: Tokenize Wikipedia"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py tokenize \
    --input data/wikipedia/wiki_text.txt \
    --output_dir data/tokenized \
    --vocab_size 10000

echo "=========================================="
echo "STEP 3: Extract DailyDialog"
echo "=========================================="
python ryan_gpt_data/extract_dailydialog.py \
    --output_dir data/dailydialog

echo "=========================================="
echo "STEP 4: Tokenize DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/train.txt \
    --output data/dailydialog/train.npy \
    --tokenizer_dir data/tokenized

PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/validation.txt \
    --output data/dailydialog/val.npy \
    --tokenizer_dir data/tokenized

echo "=========================================="
echo "STEP 5: Pretrain on Wikipedia"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/wiki_text.npy \
    --output_dir runs/pretrain \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 32 \
    --max_steps 30000 \
    --lr 3e-4 \
    --warmup_steps 1000 \
    --log_interval 100 \
    --save_interval 2500

echo "=========================================="
echo "STEP 6: Fine-tune on DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
    --train_data data/dailydialog/train.npy \
    --val_data data/dailydialog/val.npy \
    --checkpoint runs/pretrain/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 32 \
    --max_steps 3000 \
    --lr 5e-5 \
    --warmup_steps 200 \
    --log_interval 50 \
    --save_interval 500

echo "=========================================="
echo "STEP 7: Generate / Chat"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/generate.py \
    --checkpoint runs/finetune/checkpoints/ckpt_final.pt \
    --prompt "<|user|>
Hello, how are you today?
<|assistant|>
" \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_tokens 100

echo "=========================================="
echo "DONE!"
echo "=========================================="
