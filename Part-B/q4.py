import subprocess
import re
import matplotlib.pyplot as plt

# Configuration
K_VALUES = [1, 5, 10, 50, 100]
SCENARIOS = [1, 2, 3]

# Data storage
# Dictionaries will hold data mapping: Scenario -> list of times corresponding to K_VALUES
cpu_times = []
q2_kernel_times = {1: [], 2: [], 3: []}
q2_total_times = {1: [], 2: [], 3: []}
q3_kernel_times = {1: [], 2: [], 3: []}
q3_total_times = {1: [], 2: [], 3: []}

def run_command(cmd):
    """Executes a bash command and returns the stdout string."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(result.stdout)
    return result.stdout

def extract_time(output, pattern):
    """Extracts a time float value from the stdout using a regex pattern."""
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    else:
        print(f"Failed to find match for pattern '{pattern}' in output:\n{output}")
        return None

def collect_data():
    # 1. Compile everything
    print("Compiling all programs...")
    run_command(["make", "clean"])
    run_command(["make"])
    print("Compilation finished.\n")

    # 2. Run CPU only (q1)
    for k in K_VALUES:
        out = run_command(["./q1", str(k)])
        t = extract_time(out, r"Execution time for K=\d+ million:\s+([0-9.]+)\s+seconds")
        cpu_times.append(t if t else 0.0)
    
    # 3. Run Explicit Memory (q2)
    for scenario in SCENARIOS:
        for k in K_VALUES:
            out = run_command(["./q2", str(k), str(scenario)])
            kernel_t = extract_time(out, r"STRICT Kernel Execution Time:\s+([0-9.]+)\s+seconds")
            total_t = extract_time(out, r"TOTAL Round-Trip Time \(Memcpy \+ Kernel \+ Memcpy\):\s+([0-9.]+)\s+seconds")
            q2_kernel_times[scenario].append(kernel_t if kernel_t else 0.0)
            q2_total_times[scenario].append(total_t if total_t else 0.0)

    # 4. Run Unified Memory (q3)
    for scenario in SCENARIOS:
        for k in K_VALUES:
            out = run_command(["./q3", str(k), str(scenario)])
            kernel_t = extract_time(out, r"Kernel execution \(INCLUDES Host-to-Device Page Fault overhead\):\s+([0-9.]+)\s+seconds")
            total_t = extract_time(out, r"TOTAL Round-Trip Time \(H2D Faults \+ Kernel \+ D2H Faults\):\s+([0-9.]+)\s+seconds")
            q3_kernel_times[scenario].append(kernel_t if kernel_t else 0.0)
            q3_total_times[scenario].append(total_t if total_t else 0.0)

def plot_chart(title, kernel_data, total_data, cpu_data, filename):
    """Generates a chart with 2 subplots (Total Execution, Kernel Execution)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    # Styling colors/markers for the 3 scenarios
    colors = {1: 'r-*', 2: 'g-s', 3: 'b-^'}

    # Subplot 1: Total Execution Time
    ax_total = axes[0]
    ax_total.set_title("Total Round-Trip Execution Time")
    ax_total.set_xscale('log')
    ax_total.set_yscale('log')
    ax_total.set_xlabel("K (millions)")
    ax_total.set_ylabel("Execution Time (seconds) [Log Scale]")
    
    # Plot CPU baseline
    ax_total.plot(K_VALUES, cpu_data, 'k--o', label="CPU Only (q1)", linewidth=2)
    # Plot GPU traces
    for scenario in SCENARIOS:
        ax_total.plot(K_VALUES, total_data[scenario], colors[scenario], label=f"Scenario {scenario}")
        
    ax_total.legend()
    ax_total.grid(True, which="both", ls="--", alpha=0.5)

    # Subplot 2: Kernel Execution Time
    ax_kernel = axes[1]
    ax_kernel.set_title("Strict Kernel Execution Time")
    ax_kernel.set_xscale('log')
    ax_kernel.set_yscale('log')
    ax_kernel.set_xlabel("K (millions)")
    
    # Plot CPU baseline
    ax_kernel.plot(K_VALUES, cpu_data, 'k--o', label="CPU Only (q1)", linewidth=2)
    # Plot GPU traces
    for scenario in SCENARIOS:
        ax_kernel.plot(K_VALUES, kernel_data[scenario], colors[scenario], label=f"Scenario {scenario}")

    ax_kernel.legend()
    ax_kernel.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

if __name__ == "__main__":
    import os
    if not os.path.exists('./q2.cu'):
        print("Please run this script from inside the Part-B directory.")
        exit(1)
        
    collect_data()
    
    # 5. Plotting Step 2 (Explicit Memory)
    plot_chart(
        title="Step 2: Explicit Memory Transfers (cudaMemcpy)", 
        kernel_data=q2_kernel_times, 
        total_data=q2_total_times, 
        cpu_data=cpu_times, 
        filename="q4_without_unified.jpg"
    )
    
    # 6. Plotting Step 3 (Unified Memory)
    plot_chart(
        title="Step 3: Unified Memory (Implicit Page Faults)", 
        kernel_data=q3_kernel_times, 
        total_data=q3_total_times, 
        cpu_data=cpu_times, 
        filename="q4_with_unified.jpg"
    )
    print("Done! Check your folder for the two generated JPG charts.")
