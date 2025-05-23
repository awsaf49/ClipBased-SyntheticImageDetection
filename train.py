import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score

from networks.openclipnet import OpenClipLinear
from utils.processing import make_processing, add_processing_arguments

class SyntheticImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_image_size=(224, 224)):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_image_size = target_image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
            image = Image.open(img_name).convert('RGB')
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # This case should ideally not happen if transform is always provided.
                # If it does, ensure basic ToTensor conversion.
                # For robust sizing, self.transform should be configured by make_processing.
                image = transforms.ToTensor()(image)


            # Ensure image is a tensor with correct shape (safety net)
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image) # Should be redundant if transform does its job
            if image.dim() == 2:  # If grayscale, convert to RGB
                image = image.repeat(3, 1, 1)
            if image.size(0) != 3:  # Ensure 3 channels
                 # Attempt to fix channel issues, though transform should handle this.
                if image.size(0) == 1:
                    image = image.repeat(3,1,1)
                else: # if more than 3 channels, take the first 3
                    image = image[:3]


            return image, label
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a default image and label in case of error
            return torch.zeros((3, *self.target_image_size)), torch.tensor(0.0, dtype=torch.float32)

def collate_fn(batch):
    """Custom collate function to ensure consistent tensor sizes"""
    images, labels = zip(*batch)
    
    # Stack images. Assumes images are already correctly sized by the Dataset's transform.
    # If sizes mismatch, torch.stack will raise an error, indicating an upstream issue.
    try:
        images = torch.stack(images)
    except RuntimeError as e:
        # Provide more context if stacking fails due to size mismatch
        for i, img in enumerate(images):
            print(f"Image {i} in batch has size: {img.size()}")
        raise RuntimeError(f"Failed to stack images, likely due to size mismatch. Check transform pipeline. Error: {e}")

    # Stack labels
    labels = torch.stack(labels)
    
    return images, labels

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # For F1 and AUC calculation
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)  # Add squeeze() to match dimensions
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs.squeeze() > 0).float()  # Convert to binary predictions
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Collect for metrics
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
        
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_outputs = np.array(all_outputs)
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Calculate AUC - handle edge cases
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = float('nan')
    
    logger.info(f'Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 (macro): {f1:.4f}, AUC: {auc:.4f}')
    return avg_loss, accuracy, f1, auc

def validate(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # For F1 and AUC calculation
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)  # Add squeeze() to match dimensions
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0).float()  # Convert to binary predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_outputs = np.array(all_outputs)
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Calculate AUC - handle edge cases
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = float('nan')
    
    logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 (macro): {f1:.4f}, AUC: {auc:.4f}')
    return avg_loss, accuracy, f1, auc

def main():
    parser = argparse.ArgumentParser(description='Train synthetic image detection model')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--pretrain', type=str, default='clipL14commonpool', help='Pretrained model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser = add_processing_arguments(parser)
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training with args: {args}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine final image size for the model and placeholder images
    # cropSize takes precedence, then resizeSize. Default to 224x224 if neither.
    defined_output_size = None
    if hasattr(args, 'cropSize') and args.cropSize > 0:
        defined_output_size = (args.cropSize, args.cropSize)
        logger.info(f"Using cropSize: {args.cropSize} for final image dimension.")
    elif hasattr(args, 'resizeSize') and args.resizeSize > 0:
        # make_post in make_processing will CenterCrop to (resizeSize, resizeSize)
        defined_output_size = (args.resizeSize, args.resizeSize)
        logger.info(f"Using resizeSize: {args.resizeSize} for final image dimension (will be cropped to this size).")
    else:
        logger.info("Neither cropSize nor positive resizeSize specified via args. Defaulting to ensure cropSize=224 for model input consistency.")
        args.cropSize = 224 # Ensure make_processing uses this
        defined_output_size = (224, 224)
    
    # Setup data transforms using the potentially modified args
    transform = make_processing(args)
    
    # Create datasets
    train_dataset = SyntheticImageDataset(
        csv_file='data/train_set/mapping_train.csv',
        root_dir='data/train_set',
        transform=transform,
        target_image_size=defined_output_size
    )
    
    test_dataset = SyntheticImageDataset(
        csv_file='data/test_set/mapping_test.csv',
        root_dir='data/test_set',
        transform=transform,
        target_image_size=defined_output_size
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Add custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Add custom collate function
    )

    # Log dataset and loader information
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Training batch size: {args.batch_size}")
    logger.info(f"Calculated training steps per epoch: {len(train_loader)}")
    logger.info(f"Total test samples: {len(test_dataset)}")
    logger.info(f"Test batch size: {args.batch_size}")
    logger.info(f"Calculated test steps: {len(test_loader)}")

    # Create model
    model = OpenClipLinear(num_classes=1, pretrain=args.pretrain).to(device)
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    best_val_auc = 0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_f1, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_loss, val_acc, val_f1, val_auc = validate(model, test_loader, criterion, device, logger)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model - now considering multiple metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_acc_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'args': args
            }, save_path)
            logger.info(f"Saved best accuracy model with validation accuracy: {val_acc:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(args.save_dir, 'best_f1_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'args': args
            }, save_path)
            logger.info(f"Saved best F1 model with validation F1: {val_f1:.4f}")
            
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(args.save_dir, 'best_auc_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'args': args
            }, save_path)
            logger.info(f"Saved best AUC model with validation AUC: {val_auc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'args': args
            }, save_path)

if __name__ == '__main__':
    main() 