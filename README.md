# Speech-Based Depression Detection

A PyTorch implementation of depression detection using speech and wav2vec 2.0 pre-trained models, based on the methodology from the Nature Scientific Reports publication: **"Depression recognition using voice-based pre-training model"** (2024).

## Overview

This project implements an automated depression detection system using audio features extracted from speech. The system leverages the **wav2vec 2.0 Base model** (a pre-trained self-supervised speech representation model) fine-tuned with a small classifier network for binary classification of depression status.

### Key Features
- **Wav2Vec 2.0 Base Model**: Leverages Facebook's pre-trained speech representation model
- **Frozen Feature Encoder**: Uses frozen features for better generalization
- **High Accuracy**: Achieves ~96.49% accuracy and 0.9313 F1 score on test set
- **Data Preprocessing**: Intelligent voice segment merging (5 segments at a time)
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1, and ROC-AUC
- **Visualization Tools**: Built-in result visualization and analysis

## Methodology

### Architecture
1. **Feature Extraction**: wav2vec 2.0 Base model with frozen feature encoder
2. **Fine-tuning Network**:
   - Adaptive average pooling
   - Dropout (0.1)
   - Linear binary classifier
3. **Training Configuration**:
   - Learning rate: 1e-5
   - Batch size: 4
   - Gradient accumulation: 2 steps (effective batch size: 8)
   - Epochs: 10
   - Random seed: 103

### Data Processing
- **Segmentation**: Participant voice utterances are extracted from transcripts
- **Merging**: 5 consecutive segments are merged into single audio samples
- **Split**: 6:2:2 ratio (train:validation:test)
- **Audio Format**: 16 kHz sampling rate, WAV format

## Repository Structure

```
Speech-Based-Depression-Detection/
├── README.md                              # This file
├── wav2vec2_depression_detection.py       # Main training script
├── config.py                              # Configuration and logging setup
├── visualize_results.py                   # Results visualization script
└── wav2vec2_results_visualization.png     # Sample results visualization
```

## Installation

### Requirements
- Python 3.8+
- PyTorch with CUDA support (recommended)
- librosa (audio processing)
- soundfile (audio I/O)
- transformers (Hugging Face)
- scikit-learn (metrics)
- pandas & numpy
- matplotlib (visualization)
- tqdm (progress bars)

### Setup

```bash
# Clone the repository
git clone https://github.com/Sowmyamaakam/Speech-Based-Depression-Detection.git
cd Speech-Based-Depression-Detection

# Install dependencies
pip install torch transformers librosa soundfile scikit-learn pandas numpy matplotlib tqdm

# For CUDA support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Data Preparation

Prepare your DAIC-WOZ dataset with the following structure:
```
Dac-woiz/
├── {participant_id}_P/
│   ├── {participant_id}_AUDIO.wav
│   └── {participant_id}_TRANSCRIPT.csv
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
└── full_test_split.csv
```

### 2. Training

Run the main training script:

```bash
python wav2vec2_depression_detection.py \
    --data_dir Dac-woiz \
    --merged_audio_dir merged_audio \
    --output_dir models_wav2vec2 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --random_seed 103
```

#### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `Dac-woiz` | Directory containing DAIC-WOZ data |
| `--merged_audio_dir` | `merged_audio` | Directory to save merged audio files |
| `--output_dir` | `models_wav2vec2` | Directory to save models and results |
| `--batch_size` | `4` | Batch size for training |
| `--learning_rate` | `1e-5` | Learning rate for optimizer |
| `--num_epochs` | `10` | Number of training epochs |
| `--random_seed` | `103` | Random seed for reproducibility |
| `--skip_preprocessing` | `False` | Skip audio preprocessing if already done |

### 3. Visualization

Generate result visualizations:

```bash
python visualize_results.py
```

This will create `wav2vec2_results_visualization.png` showing:
- Train vs Validation Loss
- Validation Accuracy & F1 Score
- Validation ROC-AUC
- Confusion Matrix
- Per-Class F1 Scores
- Test Metrics Summary

## Outputs

### Model Checkpoints
- `models_wav2vec2/best_model.pth` - Best model checkpoint
- `models_wav2vec2/training_history.json` - Training metrics over epochs
- `models_wav2vec2/test_results.json` - Final test set metrics

### Expected Results

When trained following the paper's methodology:
- **Accuracy**: 96.49%
- **F1 Score**: 0.9313
- **Precision**: High precision on depression detection
- **Recall**: High sensitivity to depression cases
- **ROC-AUC**: 0.95+

### Log Output

Training progress is logged to:
- Console (stdout)
- `wav2vec2_depression_detection.log`

## Project Files

### `wav2vec2_depression_detection.py`
Main implementation containing:
- **VoiceSegmentMerger**: Handles audio preprocessing and segment merging
- **DepressionAudioDataset**: Custom PyTorch dataset for audio samples
- **Wav2Vec2DepressionClassifier**: Neural network model architecture
- **Wav2Vec2Trainer**: Training loop with validation and evaluation
- **main()**: Orchestrates the full pipeline

### `config.py`
Minimal configuration module providing:
- Logging setup for training and evaluation
- Centralized logger instance

### `visualize_results.py`
Comprehensive visualization script that:
- Loads training history and test results
- Creates 6-subplot figure with training curves and metrics
- Displays confusion matrix with sensitivity/specificity
- Saves high-quality PNG visualization

## Citation

This implementation is based on:

```bibtex
@article{depression_2024,
  title={Depression recognition using voice-based pre-training model},
  journal={Nature Scientific Reports},
  year={2024},
  url={https://www.nature.com/articles/s41598-024-63556-0}
}
```

## Dataset

The project uses the **DAIC-WOZ (Depression and Anxiety Interview Collection - Wizard of Oz)** dataset, which contains:
- Audio recordings of clinical interviews
- Transcripts with speaker labels
- Depression severity labels (PHQ-8 scores)
- ~140 participants with diverse depression levels

Dataset access: https://dcapswoz.ict.usc.edu/

## Features

### Audio Features
The model automatically extracts speech representations using wav2vec 2.0, capturing:
- Acoustic patterns
- Prosodic features
- Voice quality indicators
- Speech dynamics

### Model Architecture Highlights
- **Frozen Feature Encoder**: Prevents overfitting on small datasets
- **Average Pooling**: Aggregates temporal information
- **Dropout Regularization**: Reduces overfitting (p=0.1)
- **Binary Classification**: Non-depressed (0) vs. Depressed (1)

## Training Pipeline

```
Data Preparation (step 1/5)
    ↓
Data Splitting 6:2:2 (step 2/5)
    ↓
Dataset & DataLoader Creation (step 3/5)
    ↓
Model Initialization (step 4/5)
    ↓
Training & Validation (step 5/5)
    ↓
Test Evaluation & Results
```

## Reproducibility

All random seeds are controlled for reproducibility:
- NumPy seed: `args.random_seed`
- PyTorch CPU seed: `args.random_seed`
- CUDA seeds (if GPU available): `args.random_seed`

Default seed (103) matches the paper's configuration.

## Performance Metrics

The model evaluates using:
- **Accuracy**: Overall correctness
- **Precision**: True positives among positive predictions
- **Recall**: True positives among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Ability to discriminate between classes
- **Confusion Matrix**: Detailed classification results per class

## System Requirements

### Minimum
- GPU: 4GB VRAM (recommended for batch size 4)
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 50GB (for datasets and models)

### Recommended
- GPU: 12GB+ VRAM (V100 or newer)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ (for datasets)

## Troubleshooting

### Out of Memory Errors
Reduce `--batch_size` (e.g., 2 or 1)

### Missing Audio Files
Verify DAIC-WOZ dataset structure matches requirements

### Gradient Accumulation Issues
Adjust `self.accumulation_steps` in `Wav2Vec2Trainer` class

## Future Improvements

- [ ] Multi-label classification (depression severity levels)
- [ ] Transfer learning on other speech datasets
- [ ] Attention mechanisms for interpretability
- [ ] Real-time depression detection API
- [ ] Data augmentation techniques
- [ ] Ensemble methods with multiple models

## License

This project is provided as-is for research and educational purposes.

## Contact & Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Last Updated**: 2024  
**Language**: Python 3.8+  
**Framework**: PyTorch  
**Status**: Active
