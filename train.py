"""
Training Script for Deepfake Voice Detection
Uses batch generators — safe for 60,000+ audio files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
import tensorflow as tf
import keras

sys.path.append(os.path.dirname(__file__))

from src.preprocess import preprocess_audio
from src.features import (extract_mel_spectrogram, prepare_cnn_input,
                          pad_features, normalize_features)
from src.model import create_cnn_model, compile_model, get_model_summary, save_model
from src.generator import AudioDataGenerator, build_file_lists


# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir'           : 'data',
    'sample_rate'        : 16000,
    'audio_duration'     : 3.0,
    'test_size'          : 0.2,
    'validation_split'   : 0.15,
    'random_state'       : 42,
    'batch_size'         : 32,
    'epochs'             : 50,
    'learning_rate'      : 0.0001,
    'model_path'         : 'models/deepfake_detector.h5',
    'plots_dir'          : 'plots',

    # Set to None to use ALL 30k+30k files (recommended with generator)
    # Set e.g. 10000 to cap per class during quick testing
    'max_per_class'      : None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def create_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)


def plot_training_history(history, save_path='plots/training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0,0].plot(history.history['loss'],         label='Train Loss')
    axes[0,0].plot(history.history['val_loss'],     label='Val Loss')
    axes[0,0].set_title('Loss');  axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(history.history['accuracy'],     label='Train Acc')
    axes[0,1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,0].plot(history.history['precision'],    label='Train Precision')
    axes[1,0].plot(history.history['val_precision'],label='Val Precision')
    axes[1,0].set_title('Precision'); axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].plot(history.history['recall'],       label='Train Recall')
    axes[1,1].plot(history.history['val_recall'],   label='Val Recall')
    axes[1,1].set_title('Recall'); axes[1,1].legend(); axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved → {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='plots/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved → {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='plots/roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], 'navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved → {save_path}")
    plt.close()


def evaluate_model(model, test_gen, target_width):
    """Run full evaluation on the test generator."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    all_proba = []
    all_true  = []

    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        proba = model.predict(X_batch, verbose=0).flatten()
        all_proba.extend(proba)
        all_true.extend(y_batch)

    y_true       = np.array(all_true)
    y_pred_proba = np.array(all_proba)
    y_pred       = (y_pred_proba > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Real','Fake'], digits=4))

    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print("="*70)

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred_proba)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("DEEPFAKE VOICE DETECTION — GENERATOR-BASED TRAINING")
    print("="*70 + "\n")

    create_directories()
    np.random.seed(CONFIG['random_state'])
    tf.random.set_seed(CONFIG['random_state'])

    # ── Step 1: Scan files (no RAM used yet) ──────────────────────────────────
    print("Step 1: Scanning dataset files...")
    all_paths, all_labels = build_file_lists(
        CONFIG['data_dir'],
        max_per_class=CONFIG['max_per_class'],
        random_state=CONFIG['random_state']
    )

    print(f"\nTotal files : {len(all_paths):,}")
    print(f"Real        : {all_labels.count(0):,}")
    print(f"Fake        : {all_labels.count(1):,}")

    if len(all_paths) == 0:
        print("\nERROR: No audio files found!")
        print("Place files in data/real/ and data/fake/")
        return

    # ── Step 2: Train / Test split ────────────────────────────────────────────
    print("\nStep 2: Splitting into train / val / test sets...")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=all_labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state'],
        stratify=train_labels
    )

    print(f"  Train : {len(train_paths):,} files")
    print(f"  Val   : {len(val_paths):,}   files")
    print(f"  Test  : {len(test_paths):,}  files")

    # ── Step 3: Compute target spectrogram width from one sample ──────────────
    print("\nStep 3: Computing spectrogram shape from one sample...")
    sample_audio  = preprocess_audio(train_paths[0],
                                     sr=CONFIG['sample_rate'],
                                     duration=CONFIG['audio_duration'])
    sample_spec   = extract_mel_spectrogram(sample_audio, sr=CONFIG['sample_rate'])
    sample_feat   = prepare_cnn_input(sample_spec)
    target_width  = sample_feat.shape[1]
    input_shape   = sample_feat.shape          # (H, W, 1)
    print(f"  Input shape : {input_shape}")

    # ── Step 4: Build generators ──────────────────────────────────────────────
    print("\nStep 4: Building data generators...")

    train_gen = AudioDataGenerator(
        train_paths, train_labels,
        sr=CONFIG['sample_rate'],
        duration=CONFIG['audio_duration'],
        batch_size=CONFIG['batch_size'],
        target_width=target_width,
        shuffle=True
    )

    val_gen = AudioDataGenerator(
        val_paths, val_labels,
        sr=CONFIG['sample_rate'],
        duration=CONFIG['audio_duration'],
        batch_size=CONFIG['batch_size'],
        target_width=target_width,
        shuffle=False
    )

    test_gen = AudioDataGenerator(
        test_paths, test_labels,
        sr=CONFIG['sample_rate'],
        duration=CONFIG['audio_duration'],
        batch_size=CONFIG['batch_size'],
        target_width=target_width,
        shuffle=False
    )

    print(f"  Train batches : {len(train_gen):,}")
    print(f"  Val batches   : {len(val_gen):,}")
    print(f"  Test batches  : {len(test_gen):,}")

    # ── Step 5: Create & compile model ────────────────────────────────────────
    print("\nStep 5: Building model...")
    model = create_cnn_model(input_shape)
    model = compile_model(model, learning_rate=CONFIG['learning_rate'])
    get_model_summary(model)

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print("Step 6: Training...\n")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            CONFIG['model_path'],
            monitor='val_loss',
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 7: Plot history ──────────────────────────────────────────────────
    print("\nStep 7: Saving training plots...")
    plot_training_history(history)

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    print("\nStep 8: Evaluating on test set...")
    evaluate_model(model, test_gen, target_width)

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    print("\nStep 9: Saving final model...")
    save_model(model, CONFIG['model_path'])

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"  Model → {CONFIG['model_path']}")
    print(f"  Plots → {CONFIG['plots_dir']}/")
    print("\nRun the app:  streamlit run app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()