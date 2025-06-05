import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# AWS Lambda Constraints
lambda_memory_limit = 10 * 1024  # 10 GB in MB (RAM limit)
lambda_time_limit = 900  # 900 seconds (Execution Time limit)
lambda_storage_limit = 512  # 512MB max storage for deployment package

# Model Parameter Configuration
precision_size_fp32 = 4  # Each parameter in FP32 takes 4 bytes
precision_size_fp16 = 2  # Each parameter in FP16 takes 2 bytes

# Experiment Parameters
num_experts_list = [10, 20, 50, 100, 200, 500, 1000]  # Number of experts
top_k_list = [1, 2, 5, 10, 20, 50, 100]  # Top-K selected experts
param_per_expert_list = [1e6, 5e6, 10e6, 50e6, 100e6, 500e6, 1e9]  # Parameters per expert

# Storage for results
max_params_memory_constraint = np.zeros((len(num_experts_list), len(top_k_list)))
max_params_time_constraint = np.zeros((len(num_experts_list), len(top_k_list)))
max_params_storage_constraint = np.zeros((len(num_experts_list), len(top_k_list)))

# Hypothetical time per inference for a single expert (adjust based on real OpenVINO tests)
base_time_per_param = 1e-9  # Assume 1 ns per parameter for inference
base_overhead_time = 0.1  # Assume 0.1s base model overhead

# Compute max parameters that fit within AWS Lambda constraints
for i, num_experts in enumerate(num_experts_list):
    for j, top_k in enumerate(top_k_list):
        if top_k > num_experts:
            continue  # Ensure top-k is never greater than num_experts

        for param_per_expert in param_per_expert_list:
            # Memory Calculation (FP32)
            total_memory = top_k * param_per_expert * precision_size_fp32 / (1024 ** 2)  # Convert bytes to MB

            # Time Calculation
            inference_time = base_overhead_time + (top_k * param_per_expert * base_time_per_param)

            # Storage Calculation (FP32, all experts stored on disk)
            total_storage = num_experts * param_per_expert * precision_size_fp32 / (1024 ** 2)  # Convert bytes to MB

            # Check feasibility
            if total_memory <= lambda_memory_limit:
                max_params_memory_constraint[i, j] = param_per_expert  # Store max parameters fitting in memory
            if inference_time <= lambda_time_limit:
                max_params_time_constraint[i, j] = param_per_expert  # Store max parameters fitting in time
            if total_storage <= lambda_storage_limit:
                max_params_storage_constraint[i, j] = param_per_expert  # Store max parameters fitting in storage

# Convert results to DataFrames for visualization
memory_constraint_df = pd.DataFrame(max_params_memory_constraint, index=num_experts_list, columns=top_k_list)
time_constraint_df = pd.DataFrame(max_params_time_constraint, index=num_experts_list, columns=top_k_list)
storage_constraint_df = pd.DataFrame(max_params_storage_constraint, index=num_experts_list, columns=top_k_list)

# Save results to CSV for easier inspection
memory_constraint_df.to_csv("memory_usage_results.csv")
time_constraint_df.to_csv("execution_time_results.csv")
storage_constraint_df.to_csv("storage_usage_results.csv")

# Function to plot heatmaps with values
def plot_heatmap(data, title, cmap):
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data, annot=True, fmt=".1e", cmap=cmap, linewidths=0.5)
    plt.xlabel("Top-K Selected Experts")
    plt.ylabel("Number of Experts")
    plt.title(title)
    plt.show()

# Plot each constraint with improved visualization
plot_heatmap(memory_constraint_df, "Max Parameters (Memory Constraint)", "viridis")
plot_heatmap(time_constraint_df, "Max Parameters (Time Constraint)", "plasma")
plot_heatmap(storage_constraint_df, "Max Parameters (Storage Constraint)", "coolwarm")

# Print Key Insights
print("\n=== KEY INSIGHTS ===")

# Memory Constraint Insights
max_mem_expert = memory_constraint_df.max().max()
print(f"✔ Max Parameters fitting in AWS Lambda RAM (10GB): {max_mem_expert:.1e} params per expert.")

# Time Constraint Insights
max_time_expert = time_constraint_df.max().max()
print(f"✔ Max Parameters fitting in AWS Lambda Execution Time (900s): {max_time_expert:.1e} params per expert.")

# Storage Constraint Insights
max_storage_expert = storage_constraint_df.max().max()
print(f"✔ Max Parameters fitting in AWS Lambda Storage (512MB): {max_storage_expert:.1e} params per expert.")

# Feasibility Analysis
if max_mem_expert < 1e6 or max_time_expert < 1e6 or max_storage_expert < 1e6:
    print("❌ Large expert models (1B params per expert) are **not feasible** in AWS Lambda due to storage or RAM limits.")
else:
    print("✅ MoE with OpenVINO is feasible **if** top-K selection and expert size are optimized.")

# Additional Constraints
print("\n🔹 If using FP16 instead of FP32, you can double the number of parameters fitting in memory & storage.")
print("🔹 AWS Lambda layers can be used to offload model weights and bypass storage limits.")
print("🔹 Consider batching experts dynamically to optimize memory usage.")
