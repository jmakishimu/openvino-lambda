import openvino as ov
import numpy as np
import time
import psutil
import tracemalloc
import matplotlib.pyplot as plt
import pandas as pd

# Initialize OpenVINO Core
ie = ov.Core()

# Define Input Tensor for OpenVINO Model
input_shape = [1, 128]  # Example input with 128 features
input_node = ov.opset8.parameter(input_shape, dtype=np.float32)

# Experiment Parameters
num_experts_list = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]  # Number of experts to test
top_k_list = [1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40]  # Top-K selected experts

# Storage for results
memory_results = np.zeros((len(num_experts_list), len(top_k_list)))
compute_time_results = np.zeros((len(num_experts_list), len(top_k_list)))

# Function to measure performance
def measure_performance(model, input_data):
    process = psutil.Process()

    # Start memory tracking
    tracemalloc.start()
    mem_before = process.memory_info().rss / (1024 ** 2)  # in MB
    start_time = time.time()

    output = model(input_data)

    mem_after = process.memory_info().rss / (1024 ** 2)  # in MB
    end_time = time.time()

    # Get peak memory usage
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    tracemalloc.stop()

    memory_usage = max(mem_after - mem_before, peak_memory)
    inference_time = end_time - start_time

    return output, memory_usage, inference_time

# OpenVINO inference function
def openvino_infer(input_data, compiled_model):
    input_dict = {compiled_model.input(0): input_data}
    return compiled_model.create_infer_request().infer(input_dict)[compiled_model.output(0)]

# Iterate over different configurations
for i, num_experts in enumerate(num_experts_list):
    for j, top_k in enumerate(top_k_list):
        if top_k > num_experts:
            continue  # Ensure top-k is never greater than num_experts

        # Generate gating function and experts dynamically
        gate_weights = ov.opset8.constant(np.random.rand(128, num_experts).astype(np.float32))
        gate_scores = ov.opset8.matmul(input_node, gate_weights, transpose_a=False, transpose_b=False)
        softmax_scores = ov.opset8.softmax(gate_scores, axis=-1)

        # Top-k selection
        top_k_node = ov.opset8.topk(softmax_scores, k=top_k, axis=-1, mode="max", sort="value")
        top_k_indices = top_k_node.output(1)  # Extract indices

        # Define experts
        expert_outputs = []
        for _ in range(num_experts):
            expert_weight = ov.opset8.constant(np.random.rand(128, 64).astype(np.float32))
            expert_out = ov.opset8.matmul(input_node, expert_weight, transpose_a=False, transpose_b=False)
            expert_outputs.append(expert_out)

        # Select and aggregate top-k experts
        selected_outputs = []
        for idx in range(top_k):
            expert_idx = ov.opset8.gather(ov.opset8.concat(expert_outputs, axis=0), top_k_indices, axis=0)
            selected_outputs.append(expert_idx)

        # Sum selected experts
        final_output = selected_outputs[0]
        for idx in range(1, len(selected_outputs)):
            final_output = ov.opset8.add(final_output, selected_outputs[idx])

        # Create OpenVINO model
        moe_model = ov.Model([final_output], [input_node])
        compiled_model = ie.compile_model(moe_model, "CPU")

        # Pre-load model
        infer_request = compiled_model.create_infer_request()
        dummy_input = np.random.randn(1, 128).astype(np.float32)
        infer_request.infer({compiled_model.input(0): dummy_input})

        # Measure memory usage
        _, memory_usage, compute_time = measure_performance(lambda x: openvino_infer(x, compiled_model), dummy_input)

        # Store results
        memory_results[i, j] = memory_usage
        compute_time_results[i, j] = compute_time

# AWS Lambda Feasibility Check
lambda_memory_limit = 1024  # MB
fits_in_lambda = (memory_results < lambda_memory_limit)

print("\n🛠 **AWS Lambda Feasibility Check** 🛠")
for i, num_experts in enumerate(num_experts_list):
    for j, top_k in enumerate(top_k_list):
        if top_k > num_experts:
            continue
        status = "✅ FITS" if fits_in_lambda[i, j] else "❌ EXCEEDS LIMIT"
        print(f"Experts={num_experts}, Top-K={top_k} -> Memory: {memory_results[i, j]:.2f} MB | {status}")

# Convert results to DataFrames for visualization
memory_df = pd.DataFrame(memory_results, index=num_experts_list, columns=top_k_list)
compute_time_df = pd.DataFrame(compute_time_results, index=num_experts_list, columns=top_k_list)

# Plot Memory Usage
plt.figure(figsize=(8, 5))
for j, top_k in enumerate(top_k_list):
    plt.plot(num_experts_list, memory_results[:, j], marker='o', linestyle='-', label=f'Top-K={top_k}')
plt.xlabel("Number of Experts")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Scaling with Experts and Top-K Selection")
plt.legend()
plt.grid()
plt.show()

# Plot Computation Time
plt.figure(figsize=(8, 5))
for j, top_k in enumerate(top_k_list):
    plt.plot(num_experts_list, compute_time_results[:, j], marker='o', linestyle='-', label=f'Top-K={top_k}')
plt.xlabel("Number of Experts")
plt.ylabel("Computation Time (s)")
plt.title("Computation Time Scaling with Experts and Top-K Selection")
plt.legend()
plt.grid()
plt.show()

# Heatmap for Memory Usage
plt.figure(figsize=(8, 6))
plt.imshow(memory_results, cmap='viridis', aspect='auto')
plt.colorbar(label="Memory Usage (MB)")
plt.xticks(range(len(top_k_list)), labels=top_k_list)
plt.yticks(range(len(num_experts_list)), labels=num_experts_list)
plt.xlabel("Top-K Selected Experts")
plt.ylabel("Number of Experts")
plt.title("Memory Usage Heatmap")
plt.show()

# Heatmap for Computation Time
plt.figure(figsize=(8, 6))
plt.imshow(compute_time_results, cmap='plasma', aspect='auto')
plt.colorbar(label="Computation Time (s)")
plt.xticks(range(len(top_k_list)), labels=top_k_list)
plt.yticks(range(len(num_experts_list)), labels=num_experts_list)
plt.xlabel("Top-K Selected Experts")
plt.ylabel("Number of Experts")
plt.title("Computation Time Heatmap")
plt.show()
