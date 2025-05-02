import os
import time
import csv
import numpy as np
import subprocess
import re
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from datetime import datetime

# Valid frequencies and core counts for Jetson Nano B01
CPU_FREQS = [1479, 1228, 921, 614, 307]  # MHz
GPU_FREQS = [384, 307, 230, 153, 76]     # MHz
CPU_CORES = [1, 2, 3, 4]                 # Number of active CPU cores (Nano B01 has 4 cores)

# Paths for frequency and core control
CPU_GOV_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed"
CPU_CORE_PATH = "/sys/devices/system/cpu/cpu{}/online"
GPU_GOV_PATH = "/sys/devices/gpu.0/devfreq/57000000.gpu/governor"
GPU_FREQ_PATH = "/sys/devices/gpu.0/devfreq/57000000.gpu/userspace/set_freq"
MODEL_PATH = "warmed_up_model.pth"
LOG_FILE = "power_profile_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S'))

def run_command(command, timeout=10, retries=3):
    """Execute a shell command with retries."""
    for attempt in range(retries):
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                return stdout.strip()
            else:
                print(f"Attempt {attempt + 1} failed: Error executing '{command}': {stderr}")
        except subprocess.TimeoutExpired:
            process.terminate()
            print(f"Attempt {attempt + 1} timed out: Command '{command}' timed out after {timeout} seconds")
        except Exception as e:
            print(f"Attempt {attempt + 1} exception: {e}")
        if attempt < retries - 1:
            time.sleep(2)
    return None

def set_governors():
    """Set CPU and GPU governors to userspace."""
    run_command(f"echo userspace | sudo tee {CPU_GOV_PATH}")
    run_command(f"echo userspace | sudo tee {GPU_GOV_PATH}")
    time.sleep(1)
    cpu_gov = run_command(f"cat {CPU_GOV_PATH}") or "unknown"
    gpu_gov = run_command(f"cat {GPU_GOV_PATH}") or "unknown"
    print(f"Governors set to: CPU={cpu_gov}, GPU={gpu_gov}")

def set_cpu_cores(cores):
    """Set the number of active CPU cores (0-3, since cpu0 is always on)."""
    for i in range(4):
        state = 1 if i < cores else 0
        if i == 0 and state == 0:  # CPU0 cannot be disabled
            continue
        run_command(f"echo {state} | sudo tee {CPU_CORE_PATH.format(i)}")
    time.sleep(1)
    active_cores = sum(1 for i in range(4) if (run_command(f"cat {CPU_CORE_PATH.format(i)}") or "0") == "1")
    print(f"Set active CPU cores to {cores} (confirmed: {active_cores})")

def restore_settings():
    """Restore CPU to 1479 MHz, GPU to 384 MHz, and 4 CPU cores."""
    set_cpu_cores(4)
    set_cpu_freq(1479)
    set_gpu_freq(384)
    print("Restored to 4 CPU cores, CPU to 1479 MHz, and GPU to 384 MHz")

def set_cpu_freq(freq_mhz):
    """Set CPU frequency in kHz."""
    freq_khz = freq_mhz * 1000
    if run_command(f"echo {freq_khz} | sudo tee {CPU_FREQ_PATH}"):
        time.sleep(1)
        freq_set = run_command(f"cat {CPU_FREQ_PATH}") or "unknown"
        print(f"Set CPU freq to {freq_mhz} MHz (confirmed: {freq_set} kHz)")
    else:
        print(f"Error setting CPU freq {freq_mhz} MHz")
        exit(1)

def set_gpu_freq(freq_mhz):
    """Set GPU frequency in Hz."""
    freq_hz = freq_mhz * 1000000
    if run_command(f"echo {freq_hz} | sudo tee {GPU_FREQ_PATH}"):
        time.sleep(1)
        freq_set = run_command(f"cat {GPU_FREQ_PATH}") or "unknown"
        print(f"Set GPU freq to {freq_mhz} MHz (confirmed: {freq_set} Hz)")
    else:
        print(f"Error setting GPU freq {freq_mhz} MHz")
        exit(1)

def create_model(device):
    """Create a new MobileNetv3-Small model."""
    model = mobilenet_v3_small(pretrained=False, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, criterion, optimizer

def load_warmed_up_model(device, model_path):
    """Load warmed-up model weights."""
    model, criterion, optimizer = create_model(device)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded warmed-up model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)
    return model, criterion, optimizer

def parse_tegrastats_line(line):
    """Parse a single line of tegrastats output for power data."""
    total_match = re.search(r'POM_5V_IN (\d+)/\d+', line)
    cpu_match = re.search(r'POM_5V_CPU (\d+)/\d+', line)
    gpu_match = re.search(r'POM_5V_GPU (\d+)/\d+', line)
    
    total_power = int(total_match.group(1)) if total_match else 0
    cpu_power = int(cpu_match.group(1)) if cpu_match else 0
    gpu_power = int(gpu_match.group(1)) if gpu_match else 0
    return total_power, cpu_power, gpu_power

def sample_power(tegrastats_process):
    """Sample power using tegrastats."""
    try:
        line = tegrastats_process.stdout.readline()
        if line:
            total_power, cpu_power, gpu_power = parse_tegrastats_line(line)
            soc_power = total_power - (cpu_power + gpu_power)  # Approximate SOC as remainder
            return total_power, cpu_power, gpu_power, soc_power
        return 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Power sampling error: {e}")
        return 0.0, 0.0, 0.0, 0.0

def dnn_task(model, data, target, criterion, optimizer, device, power_samples, tegrastats_process):
    """Perform one minibatch of MobileNetv3 training with power sampling."""
    model.train()
    data, target = data.to(device), target.to(device)
    
    # Sample power before minibatch
    pre_total, pre_cpu, pre_gpu, pre_soc = sample_power(tegrastats_process)
    
    # Run minibatch
    start_time = time.time()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Sample power after minibatch
    post_total, post_cpu, post_gpu, post_soc = sample_power(tegrastats_process)
    
    # Average power samples
    avg_total = (pre_total + post_total) / 2 if pre_total > 0 and post_total > 0 else max(pre_total, post_total)
    avg_cpu = (pre_cpu + post_cpu) / 2 if pre_cpu > 0 and post_cpu > 0 else max(pre_cpu, post_cpu)
    avg_gpu = (pre_gpu + post_gpu) / 2 if pre_gpu > 0 and post_gpu > 0 else max(pre_gpu, post_gpu)
    avg_soc = (pre_soc + post_soc) / 2 if pre_soc > 0 and post_soc > 0 else max(pre_soc, post_soc)
    
    if avg_total > 0:
        power_samples.append((avg_total, avg_cpu, avg_gpu, avg_soc))
    
    return end_time - start_time

def precache_dataset(dataset_path):
    """Pre-cache dataset to page cache."""
    print("Pre-caching dataset to page cache...")
    if os.system("command -v vmtouch > /dev/null") == 0:
        run_command(f"vmtouch -t {dataset_path}/*")
    else:
        run_command(f"cat {dataset_path}/* > /dev/null")
    print("Dataset pre-cached.")

def profile_power():
    """Profile steady-state power/time for fine-tuning warmed-up MobileNetv3."""
    # Setup CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
    
    # Pre-cache dataset
    dataset_path = "./data/cifar-10-batches-py"
    precache_dataset(dataset_path)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Pre-fetch one batch for profiling
    data, target = next(iter(trainloader))
    
    # Initial warm-up and save weights
    print("Running initial warm-up (5 minibatches) and saving weights...")
    model, criterion, optimizer = create_model(device)
    tegrastats_process = subprocess.Popen(['tegrastats', '--interval', '100'], stdout=subprocess.PIPE, universal_newlines=True)
    for _ in range(5):
        dnn_task(model, data, target, criterion, optimizer, device, [], tegrastats_process)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Warmed-up model weights saved to {MODEL_PATH}")
    tegrastats_process.terminate()
    
    # Initialize CSV
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["CPU_Cores", "CPU_Freq_MHz", "GPU_Freq_MHz", "Avg_Time_s", "Avg_Total_Power_mW", "Avg_CPU_Power_mW", "Avg_GPU_Power_mW", "Avg_SOC_Power_mW"])
    
    results = []
    for cores in CPU_CORES:
        set_cpu_cores(cores)
        for cpu_freq in CPU_FREQS:
            set_cpu_freq(cpu_freq)
            for gpu_freq in GPU_FREQS:
                set_gpu_freq(gpu_freq)
                print(f"\nProfiling at Cores: {cores}, CPU: {cpu_freq} MHz, GPU: {gpu_freq} MHz")
                
                # Load warmed-up model
                model, criterion, optimizer = load_warmed_up_model(device, MODEL_PATH)
                
                # Start tegrastats
                tegrastats_process = subprocess.Popen(['tegrastats', '--interval', '100'], stdout=subprocess.PIPE, universal_newlines=True)
                
                # Per-frequency warm-up (2 minibatches)
                print("Running per-frequency warm-up (2 minibatches)...")
                for _ in range(2):
                    dnn_task(model, data, target, criterion, optimizer, device, [], tegrastats_process)
                
                # Stabilize power (5 seconds)
                print("Stabilizing power for 5 seconds...")
                stab_samples = []
                start_stab = time.time()
                while time.time() - start_stab < 5:
                    stab_samples.append(sample_power(tegrastats_process))
                    time.sleep(0.1)
                print("Per-frequency warm-up and stabilization complete.")
                
                # Run 5 minibatches for steady-state measurement
                total_time = 0
                power_samples = []
                for i in range(5):
                    exec_time = dnn_task(model, data, target, criterion, optimizer, device, power_samples, tegrastats_process)
                    total_time += exec_time
                    if power_samples:
                        total_power, cpu_power, gpu_power, soc_power = power_samples[-1]
                        print(f"Minibatch {i+1}: Time={exec_time:.3f}s, Total_Power={total_power:.1f}mW, CPU={cpu_power:.1f}mW, GPU={gpu_power:.1f}mW, SOC={soc_power:.1f}mW")
                    else:
                        print(f"Minibatch {i+1}: Time={exec_time:.3f}s, No power samples collected")
                
                # Compute averages
                avg_time = total_time / 5
                if power_samples:
                    total_powers, cpu_powers, gpu_powers, soc_powers = zip(*power_samples)
                    avg_power = sum(total_powers) / len(total_powers)
                    avg_cpu_power = sum(cpu_powers) / len(cpu_powers)
                    avg_gpu_power = sum(gpu_powers) / len(gpu_powers)
                    avg_soc_power = sum(soc_powers) / len(soc_powers)
                else:
                    avg_power = avg_cpu_power = avg_gpu_power = avg_soc_power = 0.0
                    print("Warning: No power samples collected")
                print(f"Avg Time (5 minibatches): {avg_time:.3f}s, Avg Power: {avg_power:.1f}mW, CPU={avg_cpu_power:.1f}mW, GPU={avg_gpu_power:.1f}mW, SOC={avg_soc_power:.1f}mW")
                
                # Save to results
                results.append((cores, cpu_freq, gpu_freq, avg_time, avg_power, avg_cpu_power, avg_gpu_power, avg_soc_power))
                
                # Write to CSV
                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cores, cpu_freq, gpu_freq, f"{avg_time:.3f}", f"{avg_power:.1f}", f"{avg_cpu_power:.1f}", f"{avg_gpu_power:.1f}", f"{avg_soc_power:.1f}"])
                
                tegrastats_process.terminate()
    
    print(f"Results saved to {LOG_FILE}")
    return results

if __name__ == "__main__":
    set_governors()
    try:
        profiling_results = profile_power()
        print("\nSummary:")
        for cores, cpu_f, gpu_f, t, p, cp, gp, sp in profiling_results:
            print(f"Cores: {cores}, CPU: {cpu_f} MHz, GPU: {gpu_f} MHz, Avg Time: {t:.3f}s, Avg Power: {p:.1f}mW, CPU={cp:.1f}mW, GPU={gp:.1f}mW, SOC={sp:.1f}mW")
    finally:
        restore_settings()
