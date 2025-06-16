import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

def pad_time_dimension(frames, fixed_time_steps=100):
    """
    Pad or truncate the time dimension of frames to a fixed number of time steps.
    Input: frames [time, channels, height, width] (numpy or tensor)
    Output: frames [fixed_time_steps, channels, height, width] (tensor)
    """
    if isinstance(frames, np.ndarray):
        frames = torch.tensor(frames, dtype=torch.float32) # Ensure float32 for consistency
    
    current_time_steps = frames.shape[0]
    
    if current_time_steps == fixed_time_steps:
        return frames
    elif current_time_steps > fixed_time_steps:
        return frames[:fixed_time_steps]
    else:
        padding_values = (0, 0,  # Width
                          0, 0,  # Height
                          0, 0,  # Channels
                          0, fixed_time_steps - current_time_steps) # Time
        return torch.nn.functional.pad(frames, padding_values, mode='constant', value=0.0)


def plot_loss_lr(loss_list, lr_list, save_path=None):
    """Plots loss and learning rate, appending a timestamp to saved files."""
    if not loss_list and not lr_list:
        print("No data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = None # Initialize ax2 to handle cases where it might not be created

    if loss_list:
        color = 'tab:orange'
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(loss_list, color=color, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)

    if lr_list:
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(lr_list, color=color, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Adjust plot to prevent labels from being clipped
    plt.title('Training Loss and Learning Rate')

    # Add legends if both are plotted
    lines, labels = [], []
    if loss_list:
        l, la = ax1.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(la)
    if lr_list and ax2:
        l, la = ax2.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(la)
    if lines:
        ax1.legend(lines, labels, loc='best')

    if save_path:
        # Generate a timestamp string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- Save Plot ---
        plot_filename = f"loss_lr_plot_{timestamp}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # --- Save Data ---
        data_to_save = {}
        if loss_list:
            data_to_save['loss'] = loss_list
        if lr_list:
            data_to_save['learning_rate'] = lr_list
        
        df = pd.DataFrame({key: pd.Series(value) for key, value in data_to_save.items()})
        csv_filename = f"loss_lr_data_{timestamp}.csv"
        csv_path = os.path.join(save_path, csv_filename)
        df.to_csv(csv_path, index_label='Iterations')
        print(f"Data saved to {csv_path}")
        
        plt.close(fig) # Close the figure to free up memory

    else:
        plt.show()

def setup_gradient_log_file(log_dir, filename):
    """Initializes the gradient log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    try:
        with open(log_path, 'w') as f_log:
            f_log.write("Gradient Norm Log\n")
            f_log.write("====================\n")
        return log_path
    except IOError as e:
        print(f"Error: Could not initialize gradient log file {log_path}. Error: {e}")
        return None

def get_device(config_device_str):
    if config_device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device_str)