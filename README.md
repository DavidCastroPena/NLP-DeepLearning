# Project Overview

In this project, we reproduced the Generative Pre-trained Transformer 2 (GPT-2) model and fine-tuned it for various downstream tasks, introducing task-specific improvements to both the model architecture and fine-tuning techniques. We implemented GPT-2 from scratch, loaded pretrained weights from Hugging Face, and fine-tuned the model for sentiment analysis, paraphrase detection, and sonnet generation.

## Paraphrase Detection

We explored Impossible Distillation, a novel knowledge distillation framework that enables high-quality paraphrase detection using a weak teacher model. Unlike conventional distillation approaches that rely on large-scale models like GPT-3, Impossible Distillation leverages GPT-2’s paraphrastic proximity property to iteratively generate high-quality paraphrase pairs through self-distillation. Our results show that models trained with Impossible Distillation outperform both standard GPT-2 fine-tuning and ChatGPT-based distillation on paraphrase detection tasks.

## Sonnet Generation

We evaluated various decoding strategies, including greedy search, top-k sampling, and beam search. Furthermore, we explored parameter-efficient fine-tuning (PEFT) techniques, specifically LoRA, to enhance generation quality while minimizing computational overhead. Alongside these methods, we also investigated integrating Sparse Attention into the generative pretraining using Poetry Foundation Corpus for improved performances in sonnet generation. Our results show that additional pretraining on a poetry-rich dataset before fine-tuning could help the model internalize rhyme, meter, and poetic constraints more effectively.

# References

1. Jaehun Jung, Peter West, Liwei Jiang, Faeze Brahman, Ximing Lu, Jillian Fisher, Taylor Sorensen, and Yejin Choi. "Impossible distillation: From low-quality models to high-quality data." arXiv preprint arXiv:2305.16635, 2023. [Access here](https://arxiv.org/abs/2305.16635).
2. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." [Access here](https://arxiv.org/abs/2106.09685).
3. Kaggle community. "Poetry Foundation Poems." [Access here](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems/data).

