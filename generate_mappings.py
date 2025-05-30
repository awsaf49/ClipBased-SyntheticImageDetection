import os
import pandas as pd
from pathlib import Path

def create_mapping_csv(input_csv, output_csv, root_dir, is_train=True):
    """
    Create a mapping CSV file with columns: filename,label
    For training: real images (label=0) are from coco2017, synthetic images (label=1) are from coco_latent_t2i
    For testing: real images (label=0) are from real_* directories, synthetic images (label=1) are from other directories
    """
    df = pd.read_csv(input_csv)
    
    # Create new dataframe for mapping
    mapping_data = []
    
    if is_train:
        # Training set: coco2017 (real) vs coco_latent_t2i (synthetic)
        for _, row in df.iterrows():
            # Add real image (label=0)
            real_path = os.path.join(root_dir, row['filename0'])
            if os.path.exists(real_path):
                mapping_data.append({
                    'filename': row['filename0'],
                    'label': 0
                })
            
            # Add synthetic image (label=1)
            synth_path = os.path.join(root_dir, row['filename1'])
            if os.path.exists(synth_path):
                mapping_data.append({
                    'filename': row['filename1'],
                    'label': 1
                })
    else:
        # Test set: real_* directories (real) vs other directories (synthetic)
        test_dir = root_dir

        test_df = pd.read_csv(os.path.join(test_dir, 'list_test.csv'))
        for _, row in test_df.iterrows():
            mapping_data.append({
                'filename': row['filename'],
                'label': 0 if 'real' in row['typ'] else 1,
                 'category': row['typ'].split('@')[0],
            })
        
        # # Process real images (label=0)
        # for real_dir in test_dir.glob('real_*'):
        #     if real_dir.is_dir():
        #         for img_path in real_dir.rglob('*'):
        #             if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        #                 rel_path = os.path.relpath(img_path, root_dir)
        #                 mapping_data.append({
        #                     'filename': rel_path,
        #                     'label': 0
        #                 })
        
        # # Process synthetic images (label=1)
        # for synth_dir in test_dir.glob('*'):
        #     if synth_dir.is_dir() and not synth_dir.name.startswith('real_'):
        #         for img_path in synth_dir.rglob('*'):
        #             if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        #                 rel_path = os.path.relpath(img_path, root_dir)
        #                 mapping_data.append({
        #                     'filename': rel_path,
        #                     'label': 1
        #                 })
    
    # Create and save mapping dataframe
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(mapping_df)} entries")
    print(f"Class distribution:\n{mapping_df['label'].value_counts()}")

def main():
    # Create mappings for training set
    create_mapping_csv(
        input_csv='data/train_set/list_train.csv',
        output_csv='data/train_set/mapping_train.csv',
        root_dir='data/train_set',
        is_train=True
    )
    
    # Create mappings for test set
    create_mapping_csv(
        input_csv='data/test_set/list_test.csv',
        output_csv='data/test_set/mapping_test.csv',
        root_dir='data/test_set',
        is_train=False
    )

if __name__ == '__main__':
    main() 