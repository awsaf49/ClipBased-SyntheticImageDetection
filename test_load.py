import torch
import os

weight_dirs = [d for d in os.listdir('weights') if os.path.isdir(os.path.join('weights', d))]

for weight_dir in weight_dirs:
    config_path = os.path.join('weights', weight_dir, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Skipping {weight_dir} - no config.yaml")
        continue
        
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    weights_file = os.path.join('weights', weight_dir, config.get('weights_file', 'weights.pth'))
    
    print(f"\nTrying to load model from {weights_file}...")
    try:
        data = torch.load(weights_file, map_location='cpu', weights_only=False)
        print(f'Success! Keys in model: {list(data.keys())}')
    except Exception as e:
        print(f'Error: {type(e).__name__} {e}') 