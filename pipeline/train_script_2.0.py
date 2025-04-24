# Code for Step 3: Train the model. Train the efficient net model using the spectograms and manifests


import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gc
import time
import json
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from io import BytesIO
import boto3


class BirdSpeciesDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, specs_dir, transform=None):
        self.df = pd.read_csv(manifest_path)
        self.specs_dir = specs_dir
        self.transform = transform
        
        # Map species to indices
        self.species = sorted(self.df['primary_label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species)}
        self.idx_to_label  = {i: lbl for lbl, i in self.label_to_idx.items()}

        self.s3_client = boto3.client('s3')
    
    def __len__(self):
        return len(self.df)

    def _load_npy(self, local_path_or_s3):
        """Load .npy from local path or s3://bucket/key."""
        if local_path_or_s3.startswith('s3://'):
            bucket, key = local_path_or_s3.replace('s3://', '').split('/', 1)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return np.load(BytesIO(obj['Body'].read()))
        return np.load(local_path_or_s3)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        label   = self.label_to_idx[row['primary_label']]
        key   = row['s3_key']
    
        # Always resolve to the local path under /opt/ml/processing
        local_path = os.path.join(self.specs_dir, key)
    
        try:
            spec = self._load_npy(local_path)
            spec = torch.from_numpy(spec).float()
            spec = spec.permute(2, 0, 1)
            if spec.max() > 1:
                spec = spec / 255.0
            if self.transform:
                spec = self.transform(spec)
            return spec, label
        except Exception as e:
            print(f"[WARN] sample {idx} failed: {e}")
            dummy = torch.zeros((3, 128, 157))
            return dummy, label


def train_model(model, train_loader, val_loader, num_epochs=10, accumulation_steps=2):
    """
    Training function with improved stability and memory management.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        accumulation_steps: Number of batches to accumulate gradients over
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear memory before starting because we were having memory issues, pytorch can be a pain
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Move model to device
    model = model.to(device)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # For early stopping
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    
    # For saving progress
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'time_per_epoch': []
    }
    
    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # ----- Training Phase -----
            model.train()
            train_loss = 0.0
            total_samples = 0
            
            # Use tqdm for progress tracking
            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                              desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            # Reset gradients at the beginning of each epoch
            optimizer.zero_grad()
            
            for batch_idx, (inputs, labels) in train_pbar:
                try:
                    # Move data to device
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Scale loss by accumulation steps
                    loss = loss / accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Only step and zero grad after several batches (gradient accumulation)
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Update statistics - using actual batch size
                    batch_size = inputs.size(0)
                    train_loss += loss.item() * batch_size * accumulation_steps
                    total_samples += batch_size
                    
                    # Update progress bar
                    train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    # Reduce frequency of cache clearing to improve performance
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    # Skip this batch and continue with the next one
                    continue
            
            # Don't forget to step if there are any remaining gradients
            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # Calculate average training loss using actual number of samples
            if total_samples > 0:
                train_loss = train_loss / total_samples
            
            # ----- Validation Phase -----
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_probs = []
            val_total_samples = 0
            
            # Clear memory before validation (once is sufficient)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use tqdm for validation progress
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                           desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in val_pbar:
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Track actual batch size
                        batch_size = inputs.size(0)
                        val_loss += loss.item() * batch_size
                        val_total_samples += batch_size
                        
                        # Get predictions
                        probs = torch.softmax(outputs, dim=1)
                        
                        # Move to CPU to free GPU memory
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        
                        # Update progress bar
                        val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            # Calculate average validation loss using actual samples
            if val_total_samples > 0:
                val_loss = val_loss / val_total_samples
            
            # Calculate macro ROC-AUC - using num_classes directly
            try:
                num_classes = model.classifier.out_features if hasattr(model.classifier, 'out_features') else model.fc.out_features
                val_auc = calculate_macro_auc(all_labels, all_probs, num_classes)
            except Exception as e:
                print(f"Error calculating AUC: {e}")
                val_auc = 0.0
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['time_per_epoch'].append(epoch_time)
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # Update learning rate based on validation performance
            scheduler.step(val_auc)
            
            # Save checkpoint every epoch (for recovery)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'history': history
            }, 'last_checkpoint.pth')
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break
            
            # Clear memory at end of epoch (once is sufficient)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
    
    # Load best model before returning
    try:
        model.load_state_dict(torch.load('best_model.pth'))
    except:
        print("Could not load best model. Returning current model state.")
    
    return model, history


def calculate_macro_auc(true_labels, pred_probs, num_classes):
    """
    Calculate macro-averaged ROC AUC score for multi-class classification.
    
    Args:
        true_labels: List or array of true class labels (integers)
        pred_probs: List or array of predicted probabilities for each class
                    Shape should be (n_samples, n_classes)
        num_classes: Total number of classes
    
    Returns:
        Macro-averaged AUC score
    """
    # Use top-level imports instead of reimporting
    
    # Binarize the labels (one-hot encoding)
    y_true = label_binarize(true_labels, classes=range(num_classes))
    
    # Calculate macro AUC
    try:
        # For multi-class problems, use 'ovr' (one-vs-rest) approach
        macro_auc = roc_auc_score(y_true, pred_probs, average='macro', multi_class='ovr')
        return macro_auc
    except ValueError as e:
        # Handle case where some classes might not be present in the validation set
        print(f"Warning in AUC calculation: {e}")
        # Find which classes are actually present in this batch
        valid_classes = np.unique(true_labels)
        
        # Filter predictions and true values to include only classes present in this batch
        filtered_true = y_true[:, valid_classes]
        filtered_pred = np.array(pred_probs)[:, valid_classes]
        
        # Calculate AUC only for present classes
        return roc_auc_score(filtered_true, filtered_pred, average='macro', multi_class='ovr')


def create_model(num_classes):
    """
    Create a model for bird species classification using EfficientNet.
    
    Args:
        num_classes: Number of species to classify
        
    Returns:
        Initialized model
    """
    # Load EfficientNet-B0 with pretrained weights
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    
    # Modify the classifier head for our number of classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model



# Entry Point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data locations
    parser.add_argument('--train-manifest', type=str, default='/opt/ml/processing/input/train/train_manifest.csv')
    parser.add_argument('--val-manifest', type=str, default='/opt/ml/processing/input/validation/val_manifest.csv')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    
    args = parser.parse_args()

    # Note: TransposeChannels is removed as it's not needed with our dataset implementation

    # Define data transformations (now assume data is already in (3, 128, width) format)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset

    specs_dir = os.environ["SM_CHANNEL_SPECS"]
    train_dataset = BirdSpeciesDataset(args.train_manifest, specs_dir, transform=train_transform)
    val_dataset = BirdSpeciesDataset(args.val_manifest, specs_dir, transform=val_transform)
    
    # Create the custom collate function for variable-width spectrograms
    def variable_width_collate(batch):
        """
        Custom collate function that handles spectrograms with variable widths.
        Pads all spectrograms in a batch to the width of the widest one.
        
        Args:
            batch: List of tuples (spectrogram, label)
        
        Returns:
            Tuple of (batched_spectrograms, batched_labels)
        """
        # Separate spectrograms and labels
        specs, labels = zip(*batch)
        
        # Find the maximum width in this batch
        max_width = max([spec.shape[2] for spec in specs])
        
        # Create empty batch tensor with max dimensions
        batched_specs = []
        
        # Pad each spectrogram to the maximum width
        for spec in specs:
            if spec.shape[2] < max_width:
                # Calculate padding needed (only on the width dimension)
                padding_width = max_width - spec.shape[2]
                # Pad the spectrogram (pad last dim: [left, right, top, bottom])
                padded_spec = F.pad(spec, (0, padding_width, 0, 0))
                batched_specs.append(padded_spec)
            else:
                batched_specs.append(spec)
        
        # Stack the padded spectrograms into a batch
        batched_specs = torch.stack(batched_specs, dim=0)
        batched_labels = torch.tensor(labels, dtype=torch.long)
        
        return batched_specs, batched_labels
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4,
        collate_fn=variable_width_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        collate_fn=variable_width_collate
    )
    
    # Create model
    num_classes = len(train_dataset.species)
    model = create_model(num_classes)
    
    # Train model
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps
    )

    model_dir = '/opt/ml/model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and training history
    model_path = os.path.join(model_dir, 'best_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    
    # Save species mapping for inference - correctly using idx_to_label
    species_mapping = train_dataset.idx_to_label
    with open(os.path.join(model_dir, 'species_mapping.json'), 'w') as f:
        json.dump(species_mapping, f)
    
    print(f"Training completed. Model saved to {model_path}")