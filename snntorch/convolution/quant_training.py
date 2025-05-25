import extra_fun
from extra_fun import OutputManager
import torch, os, time
from torch.utils.data import DataLoader
from quant_nets import QuantizedNMNISTV2
from tonic import DiskCachedDataset

import warnings
warnings.filterwarnings("ignore", message="Named tensors and all their associated APIs are an experimental feature")

pp = extra_fun.Printing.statmsg
ppc = extra_fun.Printing.statmsg_with_counter

def load_nmnist(
    data_path: str = "./data/nmnist",
    cache_path_train: str = "./cache/train",
    cache_path_test: str = "./cache/test",
    batch_size: int = 64,
    num_workers_train: int = 6,
    num_workers_test: int = 4,
    transform: any = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Load the NMNIST dataset with specified parameters.

    Args:
        data_path (str): Path to the NMNIST dataset.
        cache_path_train (str): Path to cache training data.
        cache_path_test (str): Path to cache testing data.
        batch_size (int): Batch size for data loading.
        num_workers_train (int): Number of workers for training data loading.
        num_workers_test (int): Number of workers for testing data loading.
        transform (callable, optional): Transformations to apply to the dataset.

    Returns:
        tuple: Training and testing DataLoader objects.
    """
    
    msg_no, msg_amount = 1 , 4

    # Check if the tonic package is installed
    try:
        import tonic
        ppc("Tonic installed", msg_no=msg_no, msg_amount=msg_amount, kind="success")
        
    except ImportError:
        pp("tonic package is not installed. Please install it to load NMNIST dataset.", kind="error")
        return None, None
    
    # Check if we can access datasets
    try:
        trainset = tonic.datasets.NMNIST(
            save_to=data_path,
            transform=transform,
            train=True
        )
        testset = tonic.datasets.NMNIST(
            save_to=data_path,
            transform=transform,
            train=False
        )
        ppc("NMNIST dataset loaded successfully", msg_no=msg_no+1, msg_amount=msg_amount, kind="success")
    except Exception as e:
        pp(f"Error loading NMNIST dataset: {e}", kind="error")
        return None, None
    
    # Disk cache for training and testing datasets
    try:
        
        disk_cached_train = DiskCachedDataset(
            trainset,
            cache_path=cache_path_train,
        )
        disk_cached_test = DiskCachedDataset(
            testset,
            cache_path=cache_path_test,
        )
        ppc("Disk cache created for training and testing datasets", msg_no=msg_no+2, msg_amount=msg_amount, kind="success")
    except Exception as e:
        pp(f"Error creating disk cache: {e}", kind="error")
        return None, None


    try: 
        train_loader = DataLoader(
            disk_cached_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers_train,
            collate_fn=tonic.collation.PadTensors()
        )
        test_loader = DataLoader(
            disk_cached_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers_test,
            collate_fn=tonic.collation.PadTensors()
        )
        ppc("Data loaders created successfully", msg_no=msg_no+3, msg_amount=msg_amount, kind="success")
    except Exception as e:
        pp(f"Error creating data loaders: {e}", kind="error")
        return None, None
    
    return train_loader, test_loader

def train(
        model,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int = 0,
        total_epochs: int = 10,
):
    model.train()
    running_loss, running_correct, running_total = 0, 0, 0

    t0 = time.time()
    batch_times = []

    train_batch_losses = []
    train_batch_accuracies = []

    for batch_idx, (frames, targets) in enumerate(loader):

        batch_start_time = time.time()

        frames, targets = frames.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_loss = loss.item()
        batch_acc = (preds == targets).float().mean().item()

        running_loss += batch_loss * frames.size(0)
        running_correct += (preds == targets).sum().item()
        running_total += frames.size(0)

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        train_batch_losses.append(batch_loss)
        train_batch_accuracies.append(batch_acc)

        if (batch_idx +1) % 10 == 0:
            avg_acc = running_correct / running_total
            avg_batch_time = sum(batch_times) / len(batch_times)
            avg_loss = running_loss / running_total

            # estimate remaining time for the epoch
            remaining_batches = len(loader) - (batch_idx + 1)
            remaining_time = remaining_batches * avg_batch_time
            remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

            pp(f"Epoch [{epoch}/{total_epochs}], Batch [{batch_idx+1}/{len(loader)}], "
               f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
               f"Batch Time: {avg_batch_time:.4f}s, "
               f"Estimated Remaining Time: {remaining_time_str}", kind="info")
            
    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total
    epoch_time = time.time() - t0
    epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(epoch_time))

    

    pp(f"Epoch [{epoch+1}/{total_epochs}] completed. "
       f"Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, "
       f"Total Time: {epoch_time_str}", kind="success")
    
    return avg_loss, avg_acc, train_batch_losses, train_batch_accuracies

def test(
        model,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        epoch: int = 0,

):
    model.eval()
    total_correct, total_samples = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (frames, targets) in enumerate(loader):
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)

            batch_loss = loss.item()
            batch_acc = (preds == targets).float().mean().item()

            total_correct += (preds == targets).sum().item()
            total_samples += frames.size(0)
            running_loss += batch_loss * frames.size(0)

            if (batch_idx + 1) % 10 == 0:
                ppc(f"Testing | Batch accuracy: {batch_acc:.4f} | Batch loss: {batch_loss:.4f}", msg=batch_idx + 1, msg_amount=len(loader), kind="test")
                
    avg_loss = running_loss / total_samples
    avg_acc = total_correct / total_samples

    pp(f"Testing completed. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}", kind="success")
    return avg_acc, avg_loss

def save_checkpoint(model, optimizer, epoch, train_losses, train_accuracies, test_losses, test_accuracies, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
    torch.save(checkpoint, filename)


if __name__ == "__main__":
    pp("Initializing training script for quantized training", kind="step")
    msg_no, msg_amount = 1, 8
    ppc("Loading NMNIST dataset", msg_no=msg_no, msg_amount=msg_amount, kind="info")
    # make a transform for the dataset
    import tonic.transforms as transforms
    # we only need the positive spikes at first, lets make a smol function
    class OnlyPositive(object):
        def __call__(self, frames):
            # Input: frames [T, 2, H, W] (polarity in dim=1)
            # Output: only positive polarity (assumes polarity index 1 is positive in tonic NMNIST)
            return frames[:, 1:2, :, :]

    # Define the transform
    frame_transform = transforms.Compose([
        transforms.ToFrame(sensor_size=(34, 34), time_window=1000),
        OnlyPositive()
    ])

    # Load NMNIST dataset
    train_loader, test_loader = load_nmnist(
        data_path="./data/nmnist",
        cache_path_train="./cache/nmnist_train",
        cache_path_test="./cache/nmnist_test",
        batch_size=32,
        num_workers_train=6,
        num_workers_test=4,
        transform=frame_transform
    )

    # Print lengths of the datasets
    if train_loader is not None and test_loader is not None:
        pp(f"Training dataset size\t {len(train_loader.dataset)}", kind="test")
        pp(f"Testing dataset size\t\t {len(test_loader.dataset)}", kind="test")
    else:
        pp("Failed to load NMNIST dataset.", kind="error")

    #load model
    try:
        ppc("Loading QuantizedConvNet model", msg_no=msg_no+1, msg_amount=msg_amount, kind="info")
        model = QuantizedNMNISTV2()
    except Exception as e:
        pp(f"Error loading QuantizedConvNet model: {e}", kind="error")
        model = None
    
    # use device cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ppc(f"Model loaded and moved to device: {device}", msg_no=msg_no+2, msg_amount=msg_amount, kind="info")
    ppc("Testing model with a sample input", msg_no=msg_no+3, msg_amount=msg_amount, kind="info")
    try:
        sample_frames, sample_targets = train_loader.dataset[0]
        pp(f"Sample frames shape: {sample_frames.shape}", kind="test")
        if sample_frames.ndim == 3:
            sample_frames = sample_frames[:, None, :, :]
        sample_frames = torch.from_numpy(sample_frames).float().unsqueeze(0).to(device)  # [1, T, 1, 34, 34]
        sample_targets = torch.tensor(sample_targets).unsqueeze(0).to(device)
        pp(f"Sample target shape: {sample_targets.shape}", kind="test")
        pp(f"Sample target value: {sample_targets.item()}", kind="test")
        with torch.no_grad():
            sample_output = model(sample_frames)
        pp(f"Sample output shape: {sample_output.shape}", kind="test")
        pp(f"Test completed successfully", kind="success")
    except Exception as e:
        pp(f"Error during sample test: {e}", kind="error")

    # Output manager
    try:
        ppc("Initializing OutputManager", msg_no=msg_no+4, msg_amount=msg_amount, kind="info")
        SaveOutput = OutputManager(output_root_dir="./output")
        pp("OutputManager initialized successfully", kind="success")
    except Exception as e:
        pp(f"Error initializing OutputManager: {e}", kind="error")
        SaveOutput = None
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ppc("Loss function and optimizer defined", msg_no=msg_no+5, msg_amount=msg_amount, kind="info")
    pp(f"Loss function: {criterion.__class__.__name__}, Optimizer: {optimizer.__class__.__name__}", kind="test")
    
    # sanity check for the train function
    try:
        ppc(f"Model sanity check with a single batch", msg_no=msg_no+6, msg_amount=msg_amount, kind="info")
        train_batch, test_batch = next(iter(train_loader)), next(iter(test_loader))

        train_frames, train_targets = train_batch[0].to(device), train_batch[1].to(device)
        test_frames, test_targets = test_batch[0].to(device), test_batch[1].to(device)

        model.train()
        optimizer.zero_grad()
        out = model(train_frames)
        loss = criterion(out, train_targets)
        loss.backward()
        optimizer.step()
        ppc(f"Train batch processed successfully", msg_no=1, msg_amount=2, kind="success")

        model.eval()
        with torch.no_grad():
            out = model(test_frames)
            test_loss = criterion(out, test_targets)
            preds = out.argmax(dim=1)
            test_acc = (preds == test_targets).float().mean().item()
        
        ppc(f"Test batch processed successfully", msg_no=2, msg_amount=2, kind="success")
    except Exception as e:
        pp(f"Error during sanity check: {e}", kind="error")
        exit(1)
    


    NUM_EPOCHS = 1
    eq = 20 * "="
    pp(f"{eq} LETS GOOOO {eq}", kind="step")
    ppc(f"Starting training for {NUM_EPOCHS} epochs", msg_no=msg_no+7, msg_amount=msg_amount, kind="info")

    for epoch in range(1, NUM_EPOCHS + 1):


        train_loss, train_acc, train_batch_losses, train_batch_accuracies = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=NUM_EPOCHS
        )

        test_acc, test_loss = test(
            model,
            test_loader,
            criterion,
            device,
            epoch=epoch
        )
        # print epoch summary
        pp(f"Epoch {epoch}/{NUM_EPOCHS} Summary: "
           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
           f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}", kind="info")

        # save checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            train_batch_losses, train_batch_accuracies,
            test_acc, test_loss,
            filename=os.path.join(SaveOutput.output_root_dir, f"checkpoint_epoch_{epoch}.pth")
        )