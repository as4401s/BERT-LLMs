# BERT-LLMs

Welcome to the BERT Exploration Repository! This repository is dedicated to exploring various aspects of BERT (Bidirectional Encoder Representations from Transformers) with a focus on visualization, transfer learning, sequence classification, and question answering tasks.

### Table of Contents

Overview
Features

### Overview

BERT, developed by Google, is a transformer-based model that has revolutionized natural language processing (NLP). It is pre-trained on a large corpus of text and can be fine-tuned for a variety of downstream tasks.

This repository provides practical implementations, visualizations, and insights into:

BERT Visualizations: Understanding BERT's attention mechanisms and embeddings.
BERT Transfer Learning: Leveraging pre-trained BERT for custom tasks.
Sequence Classification with BERT: Using BERT for tasks like sentiment analysis.
Question Answering with BERT: Building QA models with BERT.

# BERT 
---

## BERT Architecture

The architecture of BERT includes multiple transformer encoder layers. The diagram below shows how BERT processes input sequences:

![BERT Architecture](https://github.com/as4401s/BERT-LLMs/blob/main/images/0.png)

- **BERT Base:** 
  - 12 Transformer layers
  - 110M parameters
- **BERT Large:** 
  - 24 Transformer layers
  - 340M parameters
- Both models use multi-head attention and feed-forward networks.

---

## Input Embeddings

The diagram below illustrates how input embeddings are constructed in a Transformer-based model:

![Input Embedding](https://github.com/as4401s/BERT-LLMs/blob/main/images/1.png)

1. **Token Embeddings:** Represent individual tokens.
2. **Segment Embeddings:** Differentiate between two input segments.
3. **Positional Embeddings:** Encode token position in the sequence.

---

## BERT Base vs Large Model

This image compares the architecture of BERT Base and BERT Large:

![BERT Model Comparison](https://github.com/as4401s/BERT-LLMs/blob/main/images/2.png)

- Input: Tokenized sequence such as `[CLS] I love llamas`.
- Layers: 12 or 24 Transformer encoder layers process the sequence to produce contextualized word embeddings.

---

## Masked Language Modeling (MLM)

Masked Language Modeling (MLM) is a key pretraining objective used in models like BERT. It helps the model learn contextual representations of words by predicting masked tokens in a sentence. Here's how MLM works:

#### Explanation

#### Input Masking:
- A portion of the tokens in the input sequence is randomly masked. For example, in the sentence "I love [MASK] animals," the word "cute" might be masked.
- Typically, 15% of the tokens are masked:
  - 80% are replaced with the `[MASK]` token.
  - 10% are replaced with a random token.
  - 10% remain unchanged.

#### Objective:
- The model predicts the original tokens at the masked positions based on the context provided by the surrounding words.

#### Benefits:
- This forces the model to learn bidirectional context, capturing both left-to-right and right-to-left dependencies.
- MLM helps the model generalize well for downstream NLP tasks like question answering, sentiment analysis, and text classification.

The image below illustrates how MLM works in practice:

![Masked Language Modeling](https://github.com/as4401s/BERT-LLMs/blob/main/images/4.png)

1. **Input Sequence:** The sentence is tokenized, and one or more tokens are masked (e.g., `[N-MASK]` in the example).
2. **Embeddings:**
   - **Token Embeddings:** Represent the tokens in the input.
   - **Positional Embeddings:** Encode the position of each token in the sequence.
3. **Transformer Encoder:** Processes the input and predicts the masked token based on contextual information.
4. **Output:** The model predicts the masked word (e.g., "boring" in this example) using the surrounding tokens for context.

---

#### How MLM Improves Model Performance

- **Bidirectional Context:** Unlike unidirectional models, BERT can look both before and after a word to make predictions, making it highly effective at understanding context.
- **Generalization:** By pretraining on MLM, the model learns universal language representations that can be fine-tuned for specific NLP tasks.

---

### Features
Interactive Visualizations: Gain insights into BERT's inner workings, including attention heads and embeddings.
Transfer Learning Tutorials: Learn to adapt BERT for your own datasets and tasks.

### Practical Implementations:
Sequence classification (e.g., sentiment analysis).
Question answering with pre-trained BERT models.
