"""
Depression Detection using wav2vec 2.0 Pre-trained Model

This implementation follows the methodology from:
"Depression recognition using voice-based pre-training model"
Nature Scientific Reports (2024)
https://www.nature.com/articles/s41598-024-63556-0

Key methodology:
1. Segment and merge patient voice (5 segments at a time)
2. Use wav2vec 2.0 Base model with frozen feature encoder
3. Fine-tune with small network (average pooling + dropout + linear classifier)
4. Train with lr=1e-5, batch_size=4, gradient_accumulation=2, epochs=10
5. Split data 6:2:2 (train:val:test) with random seed 103

Expected results: 96.49% accuracy, 0.9313 F1 score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import soundfile as sf
from typing import List, Tuple, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import json
import pickle
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from config import logger, setup_logging


class VoiceSegmentMerger:
    """Segment and merge patient voice following the paper's methodology."""
    
    def __init__(self, data_dir: Path, output_dir: Path, segments_per_merge: int = 5):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segments_per_merge = segments_per_merge
        
        logger.info(f"Initialized VoiceSegmentMerger")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Segments per merge: {self.segments_per_merge}")
    
    def process_all_participants(self) -> pd.DataFrame:
        """Process all participants and create merged audio files.
        
        Returns:
            DataFrame with columns: audio_path, participant_id, label
        """
        logger.info("Processing all participants...")
        
        # Load labels
        train_csv = self.data_dir / "train_split_Depression_AVEC2017.csv"
        dev_csv = self.data_dir / "dev_split_Depression_AVEC2017.csv"
        test_csv = self.data_dir / "full_test_split.csv"
        
        train_df = pd.read_csv(train_csv)
        dev_df = pd.read_csv(dev_csv)
        test_df = pd.read_csv(test_csv)
        
        # Combine all
        all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
        
        # Create label dictionary (skip rows with NaN labels)
        # Note: full_test_split uses 'PHQ_Binary', train/dev use 'PHQ8_Binary'
        labels_dict = {}
        for _, row in all_df.iterrows():
            label_col = 'PHQ8_Binary' if 'PHQ8_Binary' in row.index else 'PHQ_Binary'
            if pd.notna(row[label_col]):
                labels_dict[f"{int(row['Participant_ID'])}_P"] = int(row[label_col])
        
        logger.info(f"Found {len(labels_dict)} participants")
        
        # Process each participant
        all_merged_files = []
        
        for participant_id in tqdm(sorted(labels_dict.keys()), desc="Processing participants"):
            try:
                merged_files = self.process_participant(participant_id, labels_dict[participant_id])
                all_merged_files.extend(merged_files)
            except Exception as e:
                logger.error(f"Failed to process {participant_id}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_merged_files)
        logger.info(f"Created {len(df)} merged audio files")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def process_participant(self, participant_id: str, label: int) -> List[Dict]:
        """Process a single participant.
        
        Args:
            participant_id: Participant ID (e.g., "300_P")
            label: Binary label (0 or 1)
            
        Returns:
            List of dictionaries with audio_path, participant_id, label
        """
        participant_folder = self.data_dir / participant_id
        if not participant_folder.exists():
            raise FileNotFoundError(f"Folder not found: {participant_folder}")
        
        # Load transcript
        transcript_path = participant_folder / f"{participant_id.replace('_P', '_TRANSCRIPT.csv')}"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        transcript_df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter for participant utterances only
        participant_utterances = transcript_df[transcript_df['speaker'] == 'Participant'].copy()
        
        if len(participant_utterances) == 0:
            raise ValueError(f"No participant utterances found for {participant_id}")
        
        # Load audio
        audio_path = participant_folder / f"{participant_id.replace('_P', '_AUDIO.wav')}"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract segments
        segments = []
        for _, row in participant_utterances.iterrows():
            start_time = row['start_time']
            stop_time = row['stop_time']
            
            start_sample = int(start_time * sr)
            stop_sample = int(stop_time * sr)
            
            segment = audio[start_sample:stop_sample]
            if len(segment) > 0:
                segments.append(segment)
        
        # Merge segments (5 at a time)
        merged_files = []
        for i in range(0, len(segments), self.segments_per_merge):
            merge_group = segments[i:i + self.segments_per_merge]
            
            if len(merge_group) == 0:
                continue
            
            # Concatenate segments
            merged_audio = np.concatenate(merge_group)
            
            # Save merged audio
            output_filename = f"{participant_id}_merged_{i//self.segments_per_merge:04d}.wav"
            output_path = self.output_dir / output_filename
            
            sf.write(output_path, merged_audio, sr)
            
            merged_files.append({
                'audio_path': str(output_path),
                'participant_id': participant_id,
                'label': label
            })
        
        return merged_files


class DepressionAudioDataset(Dataset):
    """Dataset for depression detection from audio."""
    
    def __init__(self, df: pd.DataFrame, processor: Wav2Vec2Processor, max_length: int = 160000):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length  # 10 seconds at 16kHz
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        audio, sr = librosa.load(row['audio_path'], sr=16000)
        
        # Pad or truncate to max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Process with wav2vec2 processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }


class Wav2Vec2DepressionClassifier(nn.Module):
    """Depression classifier using wav2vec 2.0 with frozen feature encoder.
    
    Following the paper's architecture:
    - wav2vec 2.0 Base model with frozen feature encoder
    - Average pooling
    - Dropout (0.1)
    - Linear classifier
    """
    
    def __init__(self, freeze_feature_encoder: bool = True, dropout: float = 0.1):
        super().__init__()
        
        # Load pre-trained wav2vec 2.0 Base model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze feature encoder if specified
        if freeze_feature_encoder:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Frozen wav2vec2 feature encoder parameters")
        
        # Fine-tuning network
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for Base model
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Average pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification
        
        logger.info(f"Initialized Wav2Vec2DepressionClassifier")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Freeze feature encoder: {freeze_feature_encoder}")
    
    def forward(self, input_values):
        # Extract features with wav2vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Average pooling over sequence dimension
        # Transpose to (batch, hidden_size, seq_len) for pooling
        hidden_states = hidden_states.transpose(1, 2)
        pooled = self.pooling(hidden_states).squeeze(-1)  # (batch, hidden_size)
        
        # Dropout and classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 2)
        
        return logits


class Wav2Vec2Trainer:
    """Trainer for wav2vec 2.0 depression detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-5,
        num_epochs: int = 10,
        output_dir: Path = Path("models_wav2vec2")
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer (AdamW as commonly used with transformers)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Gradient accumulation (effective batch size = 4 * 2 = 8)
        self.accumulation_steps = 2
        
        logger.info(f"Initialized Wav2Vec2Trainer")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Num epochs: {num_epochs}")
        logger.info(f"  Accumulation steps: {self.accumulation_steps}")
        logger.info(f"  Device: {device}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(input_values)
            loss = self.criterion(logits, labels)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                input_values = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_values)
                loss = self.criterion(logits, labels)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
                total_loss += loss.item()
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['roc_auc'] = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        metrics['class_0_f1'] = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
        metrics['class_1_f1'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        
        return metrics
    
    def train(self):
        """Full training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING WAV2VEC 2.0 DEPRESSION DETECTION MODEL")
        logger.info("=" * 80)
        
        best_val_f1 = 0.0
        history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            history['val_metrics'].append(val_metrics)
            
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model('best_model.pth', epoch, val_metrics)
                logger.info(f"  Saved best model (F1: {best_val_f1:.4f})")
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("\nTraining complete!")
        logger.info(f"Best validation F1: {best_val_f1:.4f}")
        
        # Load best model and evaluate on test set
        self.load_model('best_model.pth')
        test_metrics = self.evaluate(self.test_loader)
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 80)
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  Class 0 F1: {test_metrics['class_0_f1']:.4f}")
        logger.info(f"  Class 1 F1: {test_metrics['class_1_f1']:.4f}")
        logger.info(f"  Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
        
        # Save test results
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def save_model(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_model(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {filename}")


def main():
    """Main function following the paper's methodology."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wav2vec 2.0 Depression Detection')
    parser.add_argument('--data_dir', type=str, default='Dac-woiz',
                       help='Directory containing DAIC-WOZ data')
    parser.add_argument('--merged_audio_dir', type=str, default='merged_audio',
                       help='Directory to save merged audio files')
    parser.add_argument('--output_dir', type=str, default='models_wav2vec2',
                       help='Directory to save models and results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (paper uses 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (paper uses 1e-5)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs (paper uses 10)')
    parser.add_argument('--random_seed', type=int, default=103,
                       help='Random seed (paper uses 103)')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip audio preprocessing if already done')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file='wav2vec2_depression_detection.log')
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    logger.info("=" * 80)
    logger.info("WAV2VEC 2.0 DEPRESSION DETECTION")
    logger.info("Following methodology from Nature Scientific Reports (2024)")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Merged audio dir: {args.merged_audio_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Num epochs: {args.num_epochs}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    # Step 1: Preprocess audio (segment and merge)
    merged_df_path = Path(args.merged_audio_dir) / 'merged_audio_metadata.csv'
    
    if not args.skip_preprocessing or not merged_df_path.exists():
        logger.info("\n[Step 1/5] Preprocessing audio (segment and merge)...")
        merger = VoiceSegmentMerger(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.merged_audio_dir),
            segments_per_merge=5
        )
        merged_df = merger.process_all_participants()
        merged_df.to_csv(merged_df_path, index=False)
        logger.info(f"Saved merged audio metadata to {merged_df_path}")
    else:
        logger.info("\n[Step 1/5] Loading preprocessed audio metadata...")
        merged_df = pd.read_csv(merged_df_path)
        logger.info(f"Loaded {len(merged_df)} merged audio files")
    
    # Step 2: Split data (6:2:2 ratio with random seed 103)
    logger.info("\n[Step 2/5] Splitting data (6:2:2 ratio)...")
    
    # Shuffle with random seed
    merged_df = merged_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    
    n_total = len(merged_df)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    
    train_df = merged_df[:n_train]
    val_df = merged_df[n_train:n_train+n_val]
    test_df = merged_df[n_train+n_val:]
    
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    logger.info(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"  Val labels: {val_df['label'].value_counts().to_dict()}")
    logger.info(f"  Test labels: {test_df['label'].value_counts().to_dict()}")
    
    # Step 3: Create datasets and dataloaders
    logger.info("\n[Step 3/5] Creating datasets and dataloaders...")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    train_dataset = DepressionAudioDataset(train_df, processor)
    val_dataset = DepressionAudioDataset(val_df, processor)
    test_dataset = DepressionAudioDataset(test_df, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Step 4: Create model
    logger.info("\n[Step 4/5] Creating model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2DepressionClassifier(freeze_feature_encoder=True, dropout=0.1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Step 5: Train model
    logger.info("\n[Step 5/5] Training model...")
    
    trainer = Wav2Vec2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=Path(args.output_dir)
    )
    
    test_metrics = trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f} (Target: 0.9649)")
    logger.info(f"Final Test F1: {test_metrics['f1']:.4f} (Target: 0.9313)")
    
    if test_metrics['accuracy'] >= 0.90:
        logger.info("SUCCESS: Target accuracy achieved!")
    else:
        gap = 0.90 - test_metrics['accuracy']
        logger.info(f"Gap to target: {gap:.4f}")


if __name__ == "__main__":
    main()
