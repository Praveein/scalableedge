import torch
import torchvision.models as models
import subprocess
import time
import pandas as pd
import numpy as np
from setup_dataloader import setup_dataloader  # From Step 3

def profile_power_mode(model, dataloader, power_limit, gpu_clock, mem_clock, num_minibatches=40):
    """
    Profile a power mode by training the model and measuring time and power.
    Args:
        model: PyTorch model (e.g., ResNet-18)
        dataloader: DataLoader for the dataset
        power_limit (int): GPU power limit in Watts
        gpu_clock (int): GPU clock frequency in MHz
        mem_clock (int): Memory clock frequency in MHz
        num_minibatches (int): Number of minibatches to profile
    Returns:
        dict: Average minibatch time (ms) and power (mW)
    """
    # Set power mode
    if not set_power_mode(power_limit, gpu_clock, mem_clock):
        return None

    # Move model to GPU
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    # Initialize timing and power lists
    times = []
    powers = []

    # Start power monitoring in a separate process
    power_log = "power_log.csv"
    power_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv", "-lms", "1000", "-f", power_log],
        stdout=subprocess.PIPE
    )

    # Profile minibatches
    start_time = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_minibatches:
            break
        inputs, labels = inputs.to(device), labels.to(device)

        # Measure time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = model(inputs)
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        if i > 0:  # Skip first minibatch due to PyTorch profiling
            times.append(time_ms)

    # Stop power monitoring
    power_proc.terminate()
    elapsed_time = time.time() - start_time

    # Read power data
    power_data = pd.read_csv(power_log)
    power_values = power_data[" power.draw [W]"].str.replace(" W", "").astype(float) * 1000  # Convert to mW
    # Wait for power stabilization (2-3s, as per paper)
    stabilization_time = 3
    power_values = power_values[power_values.index * 1.0 >= stabilization_time]
    if len(power_values) > 0:
        avg_power = power_values.mean()
    else:
        avg_power = np.nan

    # Clean up
    subprocess.run(["rm", power_log])

    return {
        "power_limit": power_limit,
        "gpu_clock": gpu_clock,
        "mem_clock": mem_clock,
        "avg_time_ms": np.mean(times) if times else np.nan,
        "avg_power_mw": avg_power
    }

def profile_workload(power_modes_df, dataset_path, num_modes=1000):
    """
    Profile a subset of power modes for ResNet-18 on ImageNet.
    Args:
        power_modes_df: DataFrame of power modes
        dataset_path: Path to ImageNet validation dataset
        num_modes: Number of modes to profile
    Returns:
        DataFrame with profiling results
    """
    # Load model and dataloader
    model = models.resnet18(pretrained=False).cuda()
    dataloader = setup_dataloader("ResNet", dataset_path, batch_size=16, num_workers=4)

    # Sample power modes
    sampled_modes = power_modes_df.sample(n=min(num_modes, len(power_modes_df)), random_state=42)

    # Profile each mode
    results = []
    for _, row in sampled_modes.iterrows():
        result = profile_power_mode(
            model,
            dataloader,
            power_limit=row["power_limit"],
            gpu_clock=row["gpu_clock"],
            mem_clock=row["mem_clock"]
        )
        if result:
            results.append(result)
        print(f"Profiled mode: {result}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("profiling_data.csv", index=False)
    return results_df

# Example usage
if __name__ == "__main__":
    power_modes_df = pd.read_csv("power_modes.csv")
    dataset_path = "/path/to/imagenet_val"
    profiling_data = profile_workload(power_modes_df, dataset_path)