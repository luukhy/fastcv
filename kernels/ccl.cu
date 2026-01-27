#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void init_labels_kernel(const unsigned char* input,
                                    int* output,
                                    int width,
                                    int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;

    int idx = y * width + x;

    if (input[idx] > 0) {
        output[idx] = idx + 1;
    } 
    else {
        output[idx] = 0;
    }
}

__global__ void union_kernel(int* labels,
                            int width,
                            int height){

    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;
    int idx = y * width + x;
    this_label = labels[idx];  
    
}

torch::Tensor connected_components(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Input must be uint8 (byte) binary image");

    auto height = input.size(0);
    auto width = input.size(1);

    auto output = torch::zeros_like(input, torch::kInt32);

    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    init_labels_kernel<<<blocks, threads>>>(
        input.data_ptr<unsigned char>(),
        output.data_ptr<int>(),
        width,
        height
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(err));
    }

    return output;
}