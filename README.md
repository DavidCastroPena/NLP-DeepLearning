# Improving GPT-2 Task-Specific Performance via Impossible Distillation, Sparse Attention, and Parameter-Efficient Fine-Tuning

<img width="970" height="748" alt="Image" src="https://github.com/user-attachments/assets/aa1b9688-2077-4801-a51f-6668b3771c16" />


In this project, we reproduced the Generative Pre-trained Transformer 2 (GPT-2) model and fine-tuned it for various downstream tasks, introducing task-specific improvements to both the model architecture and fine-tuning techniques. We implemented GPT-2 from scratch, loaded pretrained weights from Hugging Face, and fine-tuned the model for sentiment analysis, paraphrase detection, and sonnet generation.

## Paraphrase Detection

We explored Impossible Distillation, a novel knowledge distillation framework that enables high-quality paraphrase detection using a weak teacher model. Unlike conventional distillation approaches that rely on large-scale models like GPT-3, Impossible Distillation leverages GPT-2’s paraphrastic proximity property to iteratively generate high-quality paraphrase pairs through self-distillation. Our results show that models trained with Impossible Distillation outperform both standard GPT-2 fine-tuning and ChatGPT-based distillation on paraphrase detection tasks.

## Sonnet Generation

We evaluated various decoding strategies, including greedy search, top-k sampling, and beam search. Furthermore, we explored parameter-efficient fine-tuning (PEFT) techniques, specifically LoRA, to enhance generation quality while minimizing computational overhead. Alongside these methods, we also investigated integrating Sparse Attention into the generative pretraining using Poetry Foundation Corpus for improved performances in sonnet generation. Our results show that additional pretraining on a poetry-rich dataset before fine-tuning could help the model internalize rhyme, meter, and poetic constraints more effectively.

# Experimental Details

<img width="600" height="236" alt="Image" src="https://github.com/user-attachments/assets/5505ffe5-13fc-4246-956d-f9310260a3a4" />
The experimental configurations table presents hyperparameter settings across four NLP tasks. Sentiment Classification (SST and CFIMDB datasets), Paraphrase Detection (Quora), and Sonnet Generation (Shakespearean Sonnets) all used a 1×10−5 learning rate and 10 epochs, with batch sizes varying (64/8, 16, and 8 respectively). Impossible Distillation experiments on Quora data employed a higher learning rate (1.5 × 10−5) with two approaches: Approach 1 using smaller batches (8) for fewer epochs (4) and Approach 2 using larger batches (16) for more epochs (8), indicating a methodical exploration of training efficiency trade-offs. For the sonnet generation task, we used the ’gpt’ model in all configurations—whether employing sparse attention alone, pretraining alone, or a combination of both. Specifically, we ran 10 epochs of pretraining at a batch size of 4, followed by 50 epochs of fine-tuning with a batch size of 8.

# References

1. Jaehun Jung, Peter West, Liwei Jiang, Faeze Brahman, Ximing Lu, Jillian Fisher, Taylor Sorensen, and Yejin Choi. "Impossible distillation: From low-quality models to high-quality data." arXiv preprint arXiv:2305.16635, 2023. [Access here](https://arxiv.org/abs/2305.16635).
2. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." [Access here](https://arxiv.org/abs/2106.09685).
3. Kaggle community. "Poetry Foundation Poems." [Access here](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems/data).

