# When Do Bi-LSTMs Outperform Transformers? A Data-Scale Study for Low-Resource Neural Machine Translation

This repository contains the implementation and experimental setup for our research paper investigating the comparative effectiveness of Bi-LSTM and Transformer architectures across different training data sizes in low-resource machine translation settings.

## Abstract

Transformer-based neural machine translation (NMT) models have achieved state-of-the-art results in high-resource settings, yet their performance and efficiency in genuinely low-resource scenarios remain underexplored. In this paper, we investigate the comparative effectiveness of Bi‑LSTM and Transformer architectures across a spectrum of training set sizes ranging from 10k to 200k sentence pairs. Using the IWSLT‑2017 German–English dataset, we conduct rigorous hyperparameter tuning with Optuna and evaluate models using BLEU and chrF metrics, as well as compute efficiency in BLEU per GPU-hour and per gradient step. Our results show that Bi‑LSTMs outperform Transformers in BLEU at 50k sentence pairs and have better computational efficiency for dataset sizes between 50k–100k, while Transformers regain superiority for bigger datasets. These findings highlight a crossover point in architecture efficacy and underscore the importance of data-aware model selection for low-resource machine translation. We release all code, logs, and trained models to facilitate future research.

## Key Findings

- **Crossover Point**: Bi-LSTMs outperform Transformers for moderately-small dayasets
- **Efficiency**: Bi-LSTMs achieve 244.27 BLEU/GPU-hour at 50k pairs vs. Transformer's 14.95
- **Scaling**: Transformers regain superiority at 200k pairs with 29.44 BLEU vs. 27.32 for Bi-LSTM
- **Practical Impact**: Results suggest Bi-LSTMs may be optimal for truly low-resource languages

## Project Structure

```
/
├── main.ipynb             # Main implementation notebook
├── paper/                 # Research paper source
├── data/                  # IWSLT-2017 De-En dataset and BPE-encoded versions
│   ├── bpe8k.model        # SentencePiece BPE model (8k vocab)
│   ├── train*.bpe.tsv     # Training sets of varying sizes
│   └── *.tsv              # Raw and processed data files
├── logs/                  # TensorBoard training logs
├── train/                 # Trained model checkpoints
│   ├── *_best.pt          # Best models by validation BLEU
│   └── *_final.pt         # Final checkpoints (may be overfitted)
└── tune/                  # Optuna hyperparameter optimization results
    ├── *.db               # Optuna study databases
    └── *_best.json        # Best hyperparameters per model/size
```

## Models Implemented

### Bi-LSTM Translator
- 2-layer bidirectional LSTM encoder + 2-layer unidirectional LSTM decoder
- Luong global attention mechanism
- Tunable embedding size (128-512), hidden size (256-1024), dropout, and optimization parameters

### Transformer Translator
- Standard encoder-decoder Transformer architecture
- Sinusoidal positional encoding
- Tunable model dimension (256-512), attention heads (4-8), layers (2-4), and Noam scheduler

## Experimental Setup

- **Dataset**: IWSLT-2017 German-English (downsampled to 10k, 50k, 75k, 100k, 150k, 200k pairs)
- **Tokenization**: SentencePiece BPE with 8k vocabulary
- **Hyperparameter Optimization**: Optuna with median pruning (15-25 trials per configuration)
- **Hardware**: Single NVIDIA H100 SXM GPU
- **Evaluation**: BLEU and chrF scores with beam search (beam size 4)

## Usage

### Prerequisites
```bash
pip install sacrebleu sentencepiece torch datasets==3.6.0 scipy tqdm numpy tensorboard optuna
```

### Running the Experiments
1. Open `main.ipynb` in Jupyter notebook
2. Execute cells sequentially to:
   - Download and preprocess IWSLT-2017 data
   - Train SentencePiece tokenizer
   - Run hyperparameter optimization for both models
   - Train best models and evaluate performance

### Key Configuration
- Bi-LSTM: batch size 2048, max 2000 steps or 1 GPU-hour
- Transformer: batch size 1024, max 4000 steps or 1 GPU-hour
- Warm-start optimization using best parameters from smaller datasets

## Results Summary
BLEU scores for each model and dataset are given below.

| Model | 10k | 50k | 75k | 100k | 150k | 200k |
|-------|-----|-----|-----|------|------|------|
| Bi-LSTM | 2.93 | **18.55** | 22.09 | 23.52 | 25.05 | 27.32 |
| Transformer | **7.89** | 9.71 | **25.60** | **27.53** | **28.84** | **29.44** |

**Bold** indicates superior performance at each data size.

## Efficiency Analysis

The Bi-LSTM demonstrates superior computational efficiency in mid-resource settings:
- Peak efficiency: 244.27 BLEU/GPU-hour at 50k pairs
- 3× higher BLEU/step ratio than Transformers at 50k-100k pairs
- Transformers become more efficient at larger scales (200k pairs)

## Reproducibility

All experiments use fixed random seeds and deterministic operations. The repository includes:
- Complete hyperparameter configurations for all models/sizes
- Trained model checkpoints
- TensorBoard logs for training visualization
- Optuna study databases for hyperparameter analysis

## Citation

```bibtex
@article{mahran2024bilstm,
  title={When Do Bi-LSTMs Outperform Transformers? A Data-Scale Study for Low-Resource Neural Machine Translation},
  author={Moaz Mahran, and Abdullah Shahid, and Sami Yousef},
  year={2025}
}
```

## Authors

- **Moaz Mahran** - University of Waterloo (mkgmahra@uwaterloo.ca)
- **Abdullah Shahid** - University of Waterloo (ashahi38@uwaterloo.ca)  
- **Sami Yousef** - University of Waterloo (s22youse@uwaterloo.ca)

## License

This research code is provided for academic and research purposes. Please cite our work if you use this code in your research.