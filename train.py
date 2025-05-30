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
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, recall_score
import wandb
import random

from networks.openclipnet import OpenClipLinear
from utils.processing import make_processing, add_processing_arguments

class SyntheticImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_image_size=(224, 224), load_categories=False, debug_mode=False, debug_samples=1000):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_image_size = target_image_size
        self.load_categories = load_categories
        
        # Check if category column exists
        self.has_categories = 'category' in self.data.columns
        
        # Apply debug mode if requested
        if debug_mode:
            self.data = self._apply_debug_sampling(debug_samples)

    def _apply_debug_sampling(self, debug_samples):
        """Sample balanced real/fake data for debug mode"""
        # Assuming column 1 is the label (0=real, 1=fake)
        real_data = self.data[self.data.iloc[:, 1] == 0]
        fake_data = self.data[self.data.iloc[:, 1] == 1]
        
        # Calculate samples per class (half each)
        samples_per_class = debug_samples // 2
        
        # Sample from each class
        if len(real_data) >= samples_per_class:
            sampled_real = real_data.sample(n=samples_per_class, random_state=42)
        else:
            sampled_real = real_data  # Use all available real samples
            
        if len(fake_data) >= samples_per_class:
            sampled_fake = fake_data.sample(n=samples_per_class, random_state=42)
        else:
            sampled_fake = fake_data  # Use all available fake samples
        
        # Combine and shuffle
        debug_data = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Debug mode: Using {len(debug_data)} samples ({len(sampled_real)} real, {len(sampled_fake)} fake)")
        return debug_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
            image = Image.open(img_name).convert('RGB')
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
            
            # Get category if available and requested
            category = None
            if self.load_categories and self.has_categories:
                category = self.data.iloc[idx, 2]  # Assuming category is in the 3rd column

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

            if self.load_categories and category is not None:
                return image, label, category
            else:
                return image, label
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a default image and label in case of error
            if self.load_categories:
                return torch.zeros((3, *self.target_image_size)), torch.tensor(0.0, dtype=torch.float32), "unknown"
            else:
                return torch.zeros((3, *self.target_image_size)), torch.tensor(0.0, dtype=torch.float32)

def collate_fn(batch):
    """Custom collate function to ensure consistent tensor sizes"""
    # Check if batch contains categories (3 elements per item vs 2)
    if len(batch[0]) == 3:
        images, labels, categories = zip(*batch)
        
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
        
        return images, labels, list(categories)
    else:
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
    f1 = f1_score(all_labels, all_predictions, average='macro')  # Changed from 'macro' to 'macro'
    
    # Calculate AUC - handle edge cases
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = float('nan')
    
    logger.info(f'Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 : {f1:.4f}, AUC: {auc:.4f}')
    
    # Return metrics for wandb logging in main loop
    return avg_loss, accuracy, f1, auc

# def validate(model, val_loader, criterion, device, logger):
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
    f1 = f1_score(all_labels, all_predictions, average='macro')  # Changed from 'macro' to 'macro'
    
    # Calculate AUC - handle edge cases
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = float('nan')
    
    logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 : {f1:.4f}, AUC: {auc:.4f}')
    
    # Return metrics for wandb logging in main loop
    return avg_loss, accuracy, f1, auc

def validate_by_category(model, val_loader, criterion, device, logger):
    """Validate model and return performance metrics broken down by category"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # For overall metrics
    all_labels = []
    all_predictions = []
    all_outputs = []
    
    # For category-wise metrics
    category_metrics = {}
    all_categories = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation by Category'):
            if len(batch) == 3:  # Has categories
                images, labels, categories = batch
            else:  # No categories
                images, labels = batch
                categories = ['unknown'] * len(labels)
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).sigmoid()
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()
            
            # Collect overall metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
            all_categories.extend(categories)
    
    # Calculate overall metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_outputs = np.array(all_outputs)
    all_categories = np.array(all_categories)
    
    # save all_labels, all_predictions, all_outputs to csv
    meta_df = pd.DataFrame({'labels': all_labels, 'predictions': all_predictions, 'outputs': all_outputs, 'categories': all_categories})
    meta_df.to_csv('meta_df.csv', index=False)

    avg_loss = total_loss / len(val_loader)
    # Multiply by 100 to express accuracy as a percentage (consistent with training loop)
    accuracy = balanced_accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')  # Changed from 'macro' to 'macro'
    
    try:
        auc = roc_auc_score(all_labels, all_outputs)
    except ValueError as e:
        logger.warning(f"Could not calculate overall AUC: {e}")
        auc = float('nan')
    
    # Log overall metrics
    logger.info(f'Overall Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 : {f1:.4f}, AUC: {auc:.4f}')

    wandb_metrics = {}
    # Category-wise metrics
    category_metrics = {}
    for category in list(np.unique(all_categories)):
        # if 'real' == category:
        #     continue
        # get category labels, predictions, and outputs with real category
        mask = (all_categories == category)
        cat_labels = all_labels[mask]
        cat_predictions = all_predictions[mask]
        cat_outputs = all_outputs[mask]

        # print the first 10 elements of each array
        # print("cat_labels", cat_labels)
        # print("cat_predictions", cat_predictions)
        # print("cat_outputs", cat_outputs)

        # Accuracy as a percentage for readability and consistency
        cat_accuracy = balanced_accuracy_score(cat_labels, cat_predictions)
        cat_f1 = f1_score(cat_labels, cat_predictions, average='macro')
        if category != 'real':
            cat_sensitivity = recall_score(cat_labels, cat_predictions, pos_label=1)
            cat_specificity = -1.0
        else:
            cat_specificity = recall_score(cat_labels, cat_predictions, pos_label=0)
            cat_sensitivity = -1.0
        # cat_auc = roc_auc_score(cat_labels, cat_outputs)

        category_metrics[category] = {
            # 'accuracy': cat_accuracy,
            # 'f1': cat_f1,
            # 'auc': cat_auc,
            'sensitivity': cat_sensitivity,
            'specificity': cat_specificity,
            'samples': cat_labels.shape[0]
        }
        # print("category_metrics", category_metrics)

        # wandb metrics
        # wandb_metrics[f'category/{category}/accuracy'] = cat_accuracy
        # wandb_metrics[f'category/{category}/f1'] = cat_f1
        wandb_metrics[f'category/{category}/sensitivity'] = cat_sensitivity
        wandb_metrics[f'category/{category}/specificity'] = cat_specificity
        # wandb_metrics[f'category/{category}/auc'] = cat_auc
        wandb_metrics[f'category/{category}/samples'] = cat_labels.shape[0]
        # break

    # # Get real category data for comparisons
    # real_labels = np.array(category_metrics.get('real', {}).get('labels', []))
    # real_predictions = np.array(category_metrics.get('real', {}).get('predictions', []))
    
    # Calculate and log category-wise metrics
    logger.info("\n=== Category-wise Performance ===")

    # Create DataFrame and convert to markdown
    metrics_df = pd.DataFrame(category_metrics).T
    markdown_table = metrics_df.to_markdown(index=True, tablefmt='grid')
    logger.info(f"\n{markdown_table}")
    
    # # Collect metrics for DataFrame and wandb
    # category_results = []
    # wandb_metrics = {}  # Collect all wandb metrics here
    
    # for category, metrics in category_metrics.items():
        # if metrics['total'] > 0:
        #     cat_labels = np.array(metrics['labels'])
        #     cat_predictions = np.array(metrics['predictions'])
            
        #     cat_accuracy = 100. * metrics['correct'] / metrics['total']
            
        #     # Calculate F1 for this category vs real
        #     if category == 'real':
        #         # For real category, just show accuracy (F1 not meaningful for single class)
        #         cat_f1 = float('nan')
        #     else:
        #         # For synthetic categories, calculate F1 as "real vs only this synthetic category"
        #         try:
        #             if len(real_labels) > 0:
        #                 # Create binary classification: real (0) vs this synthetic category (1)
        #                 binary_labels = np.concatenate([real_labels, cat_labels])  # real=0, synthetic=1
        #                 binary_predictions = np.concatenate([real_predictions, cat_predictions])
        #                 cat_f1 = f1_score(binary_labels, binary_predictions, average='macro')
        #             else:
        #                 cat_f1 = float('nan')
        #         except ValueError:
        #             cat_f1 = float('nan')
            
        #     # Add to results
        #     category_results.append({
        #         'Category': category,
        #         'Samples': metrics['total'],
        #         'Accuracy (%)': f"{cat_accuracy:.2f}",
        #         'F1 Score': f"{cat_f1:.4f}" if not np.isnan(cat_f1) else "N/A"
        #     })
            
            # Collect wandb metrics
            # wandb_metrics[f'category/{category}/accuracy'] = cat_accuracy
            # if not np.isnan(cat_f1):
            #     wandb_metrics[f'category/{category}/f1_score'] = cat_f1
    
    # Create DataFrame and convert to markdown
    # if category_results:
    #     df = pd.DataFrame(category_results)
    #     # Sort by category name, but put 'real' first
    #     df['sort_key'] = df['Category'].apply(lambda x: '0_real' if x == 'real' else f'1_{x}')
    #     df = df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
        
    #     # Convert to markdown and log
    #     try:
    #         markdown_table = df.to_markdown(index=False, tablefmt='grid')
    #         logger.info(f"\n{markdown_table}")
    #     except ImportError:
    #         # Fallback if tabulate is not available
    #         table_str = df.to_string(index=False)
    #         logger.info(f"\n{table_str}")
    #     except Exception:
    #         # Another fallback
    #         table_str = df.to_string(index=False)
    #         logger.info(f"\n{table_str}")
    # else:
    #     logger.info("No category metrics available")
    
    # Return metrics for combined wandb logging
    return avg_loss, accuracy, f1, auc, wandb_metrics

def set_seed(seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """
    Linear warm-up of learning rate from 0 to base_lr over warmup_epochs.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs for warm-up
        base_lr: Target learning rate after warm-up
    
    Returns:
        Current learning rate if in warm-up phase, None otherwise
    """
    if epoch < warmup_epochs:
        # Linear interpolation from 0 to base_lr
        # At epoch 0: lr = base_lr * (0+1)/warmup_epochs = base_lr/warmup_epochs
        # At epoch warmup_epochs-1: lr = base_lr * warmup_epochs/warmup_epochs = base_lr
        lr = base_lr * (epoch + 1) / warmup_epochs
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    return None

def main():
    parser = argparse.ArgumentParser(description='Train synthetic image detection model')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--pretrain', type=str, default='clipL14commonpool', help='Pretrained model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited data (1000 samples each for train/test)')
    parser.add_argument('--debug_samples', type=int, default=1000, help='Number of samples to use in debug mode')
    
    # Learning rate scheduler arguments
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warm-up epochs for cosine annealing scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
    
    # Wandb arguments
    parser.add_argument('--wandb_team', type=str, default='ece281', help='Wandb team name')
    parser.add_argument('--run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--exp_name', type=str, default='synthetic_detection', help='Experiment group name for wandb')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    
    parser = add_processing_arguments(parser)
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training with args: {args}")

    if args.debug:
        logger.info(f"DEBUG MODE ENABLED: Using {args.debug_samples} samples for train/test with balanced real/fake distribution")

    # Initialize wandb
    if not args.disable_wandb:
        wandb_config = {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'pretrain': args.pretrain,
            'debug_mode': args.debug,
            'debug_samples': args.debug_samples if args.debug else None,
            'warmup_epochs': args.warmup_epochs,
            'min_lr': args.min_lr,
            'scheduler': 'cosine_annealing_with_warmup',
        }
        
        # Generate run name if not provided
        if args.run_name is None:
            timestamp = datetime.now().strftime('%m%d_%H%M')
            args.run_name = f"{args.pretrain}_{timestamp}"
        
        wandb.init(
            project="ece281-deepfake",
            entity=args.wandb_team,
            name=args.run_name,
            group=args.exp_name,
            config=wandb_config,
            tags=["debug"] if args.debug else []
        )
        logger.info(f"Wandb initialized - Team: {args.wandb_team}, Run: {args.run_name}, Group: {args.exp_name}")
    else:
        logger.info("Wandb logging disabled")

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
        target_image_size=defined_output_size,
        load_categories=False,  # Training doesn't need categories
        debug_mode=args.debug,
        debug_samples=args.debug_samples
    )
    
    test_dataset = SyntheticImageDataset(
        csv_file='data/test_set/mapping_test.csv',
        root_dir='data/test_set',
        transform=transform,
        target_image_size=defined_output_size,
        load_categories=True,  # Test set needs categories for detailed evaluation
        debug_mode=args.debug,
        debug_samples=args.debug_samples
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
    
    # Learning rate scheduler - Cosine Annealing with warm-up
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    # Create cosine annealing scheduler (will be used after warm-up)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    logger.info(f"Learning rate scheduler: Cosine Annealing with {args.warmup_epochs} warm-up epochs")
    logger.info(f"Initial LR: {args.lr}, Min LR: {args.min_lr}, Warm-up epochs: {args.warmup_epochs}")

    # Set seed
    set_seed(args.seed if hasattr(args, "seed") else 42)

    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    best_val_auc = 0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Handle learning rate warm-up
        if epoch < args.warmup_epochs:
            current_lr = warmup_lr(optimizer, epoch, args.warmup_epochs, args.lr)
            logger.info(f"Linear warm-up phase: Epoch {epoch+1}/{args.warmup_epochs}, LR = {current_lr:.6f} ({current_lr/args.lr*100:.1f}% of target LR)")
        else:
            # Use cosine annealing scheduler after warm-up
            if epoch == args.warmup_epochs:
                logger.info("Linear warm-up completed, starting cosine annealing")
            scheduler.step()
        
        # Train
        train_loss, train_acc, train_f1, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, wandb_category_metrics = validate_by_category(model, test_loader, criterion, device, logger)
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Log to wandb immediately after validation (before learning rate update)
        if wandb.run is not None:
            wandb_start_time = datetime.now()
            
            epoch_metrics = {
                # Training metrics
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'train/f1_score': train_f1,
                'train/auc': train_auc if not np.isnan(train_auc) else None,
                
                # Validation metrics
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'val/f1_score': val_f1,
                'val/auc': val_auc if not np.isnan(val_auc) else None,
                
                # Learning rate and epoch
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            }
            
            # Add category-wise metrics
            epoch_metrics.update(wandb_category_metrics)

            
            # wandb_end_time = datetime.now()
            # wandb_duration = (wandb_end_time - wandb_start_time).total_seconds()
            # if wandb_duration > 2.0:  # Log if wandb takes more than 2 seconds
            #     logger.warning(f"Wandb logging took {wandb_duration:.2f} seconds for epoch {epoch + 1}")
            # else:
            #     logger.debug(f"Wandb logging completed in {wandb_duration:.2f} seconds for epoch {epoch + 1}")
        
        # Save best model - now considering multiple metrics
        best_metrics = {}
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc

            save_path = os.path.join(args.save_dir, args.exp_name, args.run_name, f'best_acc_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'val_auc': best_val_auc,
                'args': args
            }, save_path)
            logger.info(f"Saved best accuracy model with validation accuracy: {val_acc:.2f}%")
            
        # Log best metrics to wandb (separate from epoch metrics to avoid delays)
        if wandb.run is not None:
            best_metrics['best_val_accuracy'] = best_val_acc
            best_metrics['best_val_f1'] = best_val_f1
            best_metrics['best_val_auc'] = best_val_auc
            
            wandb.log(best_metrics, step=epoch, commit=False)

            # Log all metrics at once with explicit step and force immediate commit
            wandb.log(epoch_metrics, step=epoch, commit=True)
        
        # Save checkpoint every 10 epochs
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

    # Log final best metrics to wandb
    if wandb.run is not None:
        wandb.log({
            'final_best_accuracy': best_val_acc,
            'final_best_f1': best_val_f1,
            'final_best_auc': best_val_auc
        })
        logger.info("Training completed. Finishing wandb run...")
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 