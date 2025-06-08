import torch
import tonic
from tonic import transforms as tonic_transforms # Alias to avoid confusion with torchvision.transforms
from torchvision import transforms as torchvision_transforms
from torch.utils.data import DataLoader
from .utils import pad_time_dimension # Relative import

def get_transforms(config):
    """Builds and returns the data transforms."""
    try:
        sensor_size_raw = (128, 128, 2) # Default for DVSGesture
        if 'sensor_size' in config and config['sensor_size'] is not None:
             sensor_size_raw = tuple(config['sensor_size'])

        target_size_tuple = (config['target_width'], config['target_height'])
        
        # Create sensor_size for ToFrame by taking target H, W and original channels
        to_frame_sensor_size = (target_size_tuple[0], target_size_tuple[1], sensor_size_raw[2])

        transform_list = [
            tonic_transforms.Downsample(sensor_size=sensor_size_raw, target_size=target_size_tuple),
            tonic_transforms.ToFrame(sensor_size=to_frame_sensor_size, time_window=config['frame_length_us']),
            torchvision_transforms.Lambda(lambda x: pad_time_dimension(x, fixed_time_steps=config['n_timesteps'])),
            torchvision_transforms.Lambda(lambda x: torch.clamp(x, 0, 1).type(torch.float32)),
            torchvision_transforms.Lambda(lambda x: x[:, 1:2, :, :])  # Select only ON channel (index 1)
        ]
        return torchvision_transforms.Compose(transform_list)
    except KeyError as e:
        raise KeyError(f"Missing key in config for transforms: {e}. Needed: target_width, target_height, frame_length_us, n_timesteps")

def get_dataloaders(config):
    """Creates and returns train and test Dataloaders."""
    data_transform = get_transforms(config)
    
    data_path = config.get('data_dir', '../data') # Default data directory

    try:
        trainset = tonic.datasets.DVSGesture(save_to=data_path, train=True, transform=data_transform)
        testset = tonic.datasets.DVSGesture(save_to=data_path, train=False, transform=data_transform)
    except Exception as e:
        print(f"Error loading DVSGesture dataset from {data_path}. Ensure the path is correct and data can be downloaded/accessed.")
        print(f"Details: {e}")
        raise

    trainloader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0), # Use get for optional params
        pin_memory=True
    )
    testloader = DataLoader(
        testset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    return trainloader, testloader