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

# Next Sentence Prediction (NSP)

Next Sentence Prediction (NSP) is a key pretraining objective in BERT that helps the model understand relationships between sentences, which is vital for tasks like question answering, natural language inference, and dialog generation.

---

## Explanation

### How NSP Works:
1. **Input Pairs:**
   - The model takes two input sentences (Sentence A and Sentence B).
   - 50% of the time, Sentence B logically follows Sentence A (**"IsNext"**).
   - 50% of the time, Sentence B is a random sentence from the corpus (**"NotNext"**).

2. **Objective:**
   - The task is to predict whether Sentence B follows Sentence A.

3. **Input Representation:**
   - Sentences are tokenized and separated by a `[SEP]` token.
   - The `[CLS]` token is added at the start of the sequence.
   - Segment embeddings (A for Sentence A and B for Sentence B) differentiate the two inputs.

4. **Output:**
   - The model outputs a binary classification: **"IsNext"** or **"NotNext"**.

---

## Visual Example of NSP

While this repository doesn't currently include a visual for NSP, the process is as follows:

1. Input: `[CLS] Sentence A [SEP] Sentence B [SEP]`
2. Segment Embeddings:
   - Tokens from Sentence A: Segment A embeddings.
   - Tokens from Sentence B: Segment B embeddings.
3. Output: Binary classification to determine whether Sentence B follows Sentence A.

---

## Benefits of NSP

- **Sentence Understanding:** Helps the model capture sentence relationships and context.
- **Versatility:** NSP improves performance on NLP tasks such as question answering, summarization, and dialog systems.

---

# Differences Between BERT, DistilBERT, and RoBERTa

BERT (Bidirectional Encoder Representations from Transformers) has inspired several variants, each enhancing specific aspects of the original model. Here's a brief overview of **BERT**, **DistilBERT**, and **RoBERTa**:

---

### BERT
- **Description**: BERT is a transformer-based model developed by Google that captures bidirectional context in language understanding.
- **Pretraining Data**: 16GB of text data, including:
  - BooksCorpus
  - English Wikipedia
- **Tasks**:
  - Masked Language Modeling (MLM)
  - Next Sentence Prediction (NSP)
- **Architecture**:
  - 12 encoder layers
  - 768 hidden units
  - 110 million parameters (Base version)

---

### DistilBERT
- **Description**: DistilBERT is a compressed version of BERT, designed to be smaller, faster, and more efficient while retaining a significant portion of BERT's performance.
- **Parameters**: 66 million parameters (40% fewer than BERT)
- **Performance**: Achieves ~97% of BERT's performance.
- **Advantages**:
  - Faster training
  - Easier deployment, especially in resource-constrained environments.

---

### RoBERTa
- **Description**: RoBERTa (Robustly Optimized BERT Approach) is an optimized version of BERT, trained with improved strategies and on a larger dataset.
- **Pretraining Data**: 160GB of text data, including:
  - BooksCorpus and English Wikipedia (BERT data)
  - CommonCrawl News dataset
  - WebText
  - Stories from Common Crawl
- **Key Modifications**:
  - Removed the NSP task.
  - Used **dynamic masking** (changing masked tokens during training).
  - Trained with larger batch sizes and higher learning rates.
- **Performance**: Enhanced performance over BERT on various NLP tasks.

---

### Summary
| **Model**       | **Parameters** | **Key Features**                                                                       |
|-----------------|---------------|----------------------------------------------------------------------------------------|
| **BERT**       | 110M          | Original model capturing bidirectional context.                                        |
| **DistilBERT** | 66M           | Smaller, faster version achieving ~97% of BERT's performance.                          |
| **RoBERTa**    | 125M+         | Optimized version trained on more data with dynamic masking and improved strategies.   |

---


### Features
Interactive Visualizations: Gain insights into BERT's inner workings, including attention heads and embeddings.
Transfer Learning Tutorials: Learn to adapt BERT for your own datasets and tasks.

### Practical Implementations:
Sequence classification (e.g., sentiment analysis).
Question answering with pre-trained BERT models.
