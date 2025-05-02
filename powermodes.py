import subprocess
import pandas as pd
import itertools

def define_power_modes():
    """
    Define power modes for RTX 3080Ti based on GPU power limit, GPU clock, and memory clock.
    Returns a DataFrame with all valid power mode combinations.
    """
    # Define ranges for power modes
    power_limits = list(range(10, 41, 10))  # 100W to 350W in 10W steps
    gpu_clocks = list(range(1000, 2101, 100))  # 1000MHz to 2100MHz in 100MHz steps
    mem_clocks = list(range(5000, 9501, 500))  # 5000MHz to 9500MHz in 500MHz steps

    # Generate all combinations
    power_modes = list(itertools.product(power_limits, gpu_clocks, mem_clocks))

    # Create DataFrame
    power_modes_df = pd.DataFrame(power_modes, columns=["power_limit", "gpu_clock", "mem_clock"])

    # Filter out potentially unstable combinations (e.g., high clocks at low power)
    # Example: Remove modes where gpu_clock > 1800MHz and power_limit < 150W
    power_modes_df = power_modes_df[
        ~((power_modes_df["gpu_clock"] > 1800) & (power_modes_df["power_limit"] < 150))
    ]

    print(f"Generated {len(power_modes_df)} power modes")
    return power_modes_df

def set_power_mode(power_limit, gpu_clock, mem_clock):
    """
    Set the GPU power mode using nvidia-smi.
    Args:
        power_limit (int): Power limit in Watts
        gpu_clock (int): GPU clock frequency in MHz
        mem_clock (int): Memory clock frequency in MHz
    """
    try:
        # Set power limit
        subprocess.run(["nvidia-smi", "-pl", str(power_limit)], check=True)
        # Set GPU and memory clock (requires root privileges and persistence mode)
        subprocess.run(["nvidia-smi", "-pm", "1"], check=True)  # Enable persistence mode
        subprocess.run(["nvidia-smi", "-lgc", f"{gpu_clock},{gpu_clock}"], check=True)
        subprocess.run(["nvidia-smi", "-lmc", f"{mem_clock},{mem_clock}"], check=True)
        print(f"Set power mode: {power_limit}W, GPU {gpu_clock}MHz, Mem {mem_clock}MHz")
    except subprocess.CalledProcessError as e:
        print(f"Error setting power mode: {e}")
        return False
    return True

# Example usage
if __name__ == "__main__":
    power_modes_df = define_power_modes()
    power_modes_df.to_csv("power_modes.csv", index=False)
    # Test setting a power mode
    set_power_mode(power_limit=200, gpu_clock=1500, mem_clock=7000)