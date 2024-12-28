# CLID: Chunk-Level Intent Detection Framework for Multiple Intent Spoken Language Understanding

This repository provides the implementation of the Chunk-Level Intent Detection (CLID) framework, designed to enhance spoken language understanding (SLU) by effectively handling multiple intents within a single utterance.

## Key Features
- **Chunk-Level Detection**: Utilizes a sliding window-based self-attention mechanism to capture contextual information for intent detection.
- **Intent Transition Point Identification**: Introduces an auxiliary task to identify intent transition points, allowing for better recognition of sub-utterances with single intents.
- **Model-Agnostic**: Compatible with various pre-trained language models (e.g., BERT, RoBERTa).

## Steps to Reproduce

### Step 1: Prepare a Fine-Tuned Model
Use a well-trained classification model that has been fine-tuned on your dataset for spoken language understanding tasks.

### Step 2: Data Preparation
Prepare your dataset, ensuring it contains utterances with multiple intents. The dataset should be split into training, validation, and test sets.

### Step 3: Implement the CLID Framework
1. **Self-Attention Encoder**: Implement a self-attention encoder to capture features within token orders and contextual information.
2. **Chunk-Level Intent Detection**: Utilize the sliding window-based self-attention (SWSA) scheme to perform regional intent detection.
3. **Intent Transition Point Identification**: Implement an auxiliary task to predict intent transition points within utterances.

### Step 4: Train the Model
Train the model using multi-task learning, optimizing both intent detection and slot filling tasks.

### Step 5: Evaluate the Model
Evaluate the model on public multi-intent datasets (e.g., MixATIS, MixSNIPS) to measure performance metrics such as Intent Accuracy and Slot F1 Score.

## Installation
Clone the repository and install the required dependencies:
```
git clone https://github.com/yourusername/CLID.git
cd CLID
pip install -r requirements.txt
```

##  TRAIN
```
python train.py -g -hv -wt tf -ne --decoder agif
```

## Citation
If you use this framework in your research, please cite the following paper:
```
@inproceedings{CLID,
    title     = {{CLID: A Chunk-Level Intent Detection Framework for Multiple Intent Spoken Language Understanding}},
    author    = {H. Huang, P. Huang, Z. Zhu, J. Li and P. Lin},
    journal  = {IEEE Signal Processing Letters},
    month      ={Sep},
    year      = {2022},
    pages     = {2123 - 2127},
    volume     = {29},
    publisher     = {Institute of Electrical and Electronics Engineers (IEEE)},
    url     = {http://dx.doi.org/10.1109/LSP.2022.3211156},
}
```