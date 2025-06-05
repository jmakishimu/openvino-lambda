Deploying Llama-2 on AWS Lambda: An Optimization Study with OpenVINO

This repository investigates the feasibility of deploying the 7-billion-parameter Llama-2 model in a serverless AWS Lambda environment. The core challenge is to optimize and compress the model to fit within Lambda's strict architectural constraints, specifically its limits on deployment package size, available memory, and temporary disk storage.

This study uses Intel's OpenVINO toolkit to perform model conversion and aggressive post-training quantization (INT8 and INT4). The results quantify the reductions in storage size, memory usage, and inference latency to determine if deployment is viable.
The Challenge: Fitting a 7B Model into AWS Lambda

AWS Lambda provides a powerful serverless compute environment but imposes several key constraints that make deploying large models difficult:

    Ephemeral Storage: A limited /tmp directory space (max 10 GB). The entire model must fit here.
    Memory (RAM): A configurable memory limit, up to 10,240 MB. The model and all dependencies must load and run within this limit.
    Execution Timeout: A maximum execution time of 15 minutes, requiring inference to be fast.

The goal of this project is to use OpenVINO to shrink the Llama-2-7B model to meet these targets.
Optimization Methodology

The scripts in this repository perform the necessary optimizations to prepare the Llama-2 model for a serverless environment.
1. Model Conversion (FP16)

The baseline process involves converting the standard FP16 PyTorch model into the OpenVINO Intermediate Representation (IR) format (.xml and .bin). This step, performed by 01_fp16_conversion_demo.py, applies hardware-agnostic graph optimizations that can improve performance even before quantization.
2. Post-Training Quantization (INT8 & INT4)

To meet the size and memory constraints, the 02_quantization_and_sizing.py script applies aggressive post-training static quantization. It creates three smaller versions of the model by converting its weights to low-precision integers:

    INT8
    INT4 (Symmetric)
    INT4 (Asymmetric)

This step is critical for reducing the model's disk footprint and peak memory usage during inference.
Results: Meeting Lambda Constraints

The following results demonstrate the effectiveness of OpenVINO quantization in making the Llama-2-7B model viable for Lambda.
💾 Disk Storage (Fitting in /tmp)

The INT4-quantized model is 3.7 GB, a 72% reduction from the original 13.5 GB.

    Lambda Implication: A 3.7 GB model fits comfortably within the 10 GB /tmp ephemeral storage limit available to AWS Lambda, making deployment possible.

Model	Storage Usage (GB)
PyTorch FP16	13.5
OpenVINO FP16	13.5
OpenVINO INT8	6.8
OpenVINO INT4 (ASYM)	3.7
🧠 Peak Memory Usage (Fitting in RAM)

The INT4-quantized model requires only 5.25 GB of memory, a 62% reduction from the original ~14 GB.

    Lambda Implication: This memory footprint allows the model to be loaded and run in a Lambda function configured with an appropriate memory setting (e.g., 6144 MB or 8192 MB), which is well within the 10 GB maximum.

Model	Peak Memory Usage (GB)
PyTorch FP16	13.84
OpenVINO FP16	13.56
OpenVINO INT8	7.82
OpenVINO INT4 (ASYM)	5.25
🚀 Inference Speed (Avoiding Timeout)

The INT4-quantized model performs inference in ~6.8 seconds, more than twice as fast as the original model.

    Lambda Implication: This rapid execution time is well within Lambda's 15-minute timeout, making the model suitable for interactive, request-response workloads.

Model	Execution Time (s)
PyTorch FP16	13.92
OpenVINO FP16	11.51
OpenVINO INT8	7.64
OpenVINO INT4 (ASYM)	6.79
Conclusion

Yes, it is feasible to run the Llama-2-7B model on AWS Lambda. By leveraging OpenVINO and applying aggressive INT4 quantization, the model's storage and memory requirements can be reduced to fit within Lambda's operational limits, while also significantly improving inference speed.
How to Use

    Clone the repository and set up the environment:
    Bash

git clone <your-repo-url>
cd llama2-on-lambda-openvino
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run the experiments:
The scripts will download the base Llama-2 model and save the converted OpenVINO models in new local directories.
Bash

    # Run the FP16 conversion and performance test
    python experiments/01_fp16_conversion_demo.py

    # Generate all quantized versions and check their size
    python experiments/02_quantization_and_sizing.py

Limitations

    No Accuracy Evaluation: This study focuses on operational feasibility. The impact of INT4 quantization on model accuracy (e.g., perplexity, response quality) is not measured and would be a critical factor in a production deployment.
    Cold Starts: While inference is fast, the initial "cold start" of a Lambda function (which includes downloading the 3.7 GB model to /tmp and loading it into memory) will be significant and must be considered in application design. Provisioned Concurrency might be required.
    Hardware Dependency: Performance is dependent on the underlying Lambda execution environment hardware.

