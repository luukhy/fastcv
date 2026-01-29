#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

struct InitLabelsFunctor {
    const unsigned char* input_ptr;
    
    InitLabelsFunctor(const unsigned char* _ptr) : input_ptr(_ptr) {}

    __host__ __device__
    int operator()(const int idx) const {
        if (input_ptr[idx] > 0) {
            return idx + 1; 
        }
        return 0; 
    }
};


__global__ void union_kernel(
    int* labels,
    bool* changed,
    int width,
    int height) {

    __shared__ int s_labels[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int idx = y * width + x;

    int my_val = 0;
    if (x < width && y < height) {
        my_val = labels[idx];
    }
    s_labels[ty][tx] = my_val;
    
    __syncthreads();

    // cant return when my_val == 0 because __syncthreads wont work and will kill the program so we set a active flag 
    // and procede with iterations
    bool active = (my_val != 0); 

    int k = 0;
    
    for (int k = 0; k < 16; k++) {
        int best = s_labels[ty][tx]; 
        
        if (active) {
            if (ty > 0) {
                int neigh = s_labels[ty - 1][tx];
                if (neigh > 0) best = min(best, neigh);
            }
            if (ty < blockDim.y - 1) {
                int neigh = s_labels[ty + 1][tx];
                if (neigh > 0) best = min(best, neigh);
            }
            if (tx > 0) {
                int neigh = s_labels[ty][tx - 1];
                if (neigh > 0) best = min(best, neigh);
            }
            if (tx < blockDim.x - 1) {
                int neigh = s_labels[ty][tx + 1];
                if (neigh > 0) best = min(best, neigh);
            }
        }

        __syncthreads();

        if (active && best < s_labels[ty][tx]) {
            s_labels[ty][tx] = best;
        }
        
        __syncthreads();
    }

    if (!active) 
        return; 

    int current_label = s_labels[ty][tx];
    int new_label = current_label;

    // thread edge cases
    if (ty == 0 && y > 0) {
        int neighbor_val = labels[idx - width];
        if (neighbor_val > 0) 
            new_label = min(new_label, neighbor_val);
    }
    else if (ty == blockDim.y - 1 && y < height - 1) {
        int neighbor_val = labels[idx + width];
        if (neighbor_val > 0) 
            new_label = min(new_label, neighbor_val);
    }
    if (tx == 0 && x > 0) {
        int neighbor_val = labels[idx - 1];
        if (neighbor_val > 0) 
            new_label = min(new_label, neighbor_val);
    }
    else if (tx == blockDim.x - 1 && x < width - 1) {
        int neighbor_val = labels[idx + 1];
        if (neighbor_val > 0) 
            new_label = min(new_label, neighbor_val);
    }

    if (new_label < labels[idx]) {
        atomicMin(&labels[idx], new_label);
        *changed = true;
    }
}

__global__ void flatten_kernel(int* labels, bool* changed, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int my_label = labels[idx];
    if (my_label == 0) return;

    int parent = labels[my_label];
    
    if (parent > 0 && parent < my_label) {
        atomicMin(&labels[idx], parent);
        *changed = true;
    }
}

torch::Tensor connected_components(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Input must be uint8 (byte) binary image");

    int height = input.size(0);
    int width = input.size(1);
    int num_pixels = height * width;
    auto labels = torch::zeros({height, width}, torch::dtype(torch::kInt32).device(input.device()));

    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    thrust::transform(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_pixels),
        thrust::device_pointer_cast(labels.data_ptr<int>()),
        InitLabelsFunctor(input.data_ptr<unsigned char>())
    );

    auto changed_tensor = torch::zeros({1}, torch::dtype(torch::kBool).device(input.device()));
    bool* d_changed = changed_tensor.data_ptr<bool>();
    bool h_changed = true;

    int max_iter = width * height;
    int iter = 0;

    while (h_changed && iter < max_iter) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        union_kernel<<<blocks, threads>>>(
            labels.data_ptr<int>(),
            d_changed,
            width,
            height
        );

        int threads_flat = 256;
        int blocks_flat = (num_pixels + threads_flat - 1) / threads_flat;

        flatten_kernel<<<blocks_flat, threads_flat>>>(
            labels.data_ptr<int>(),
            d_changed,
            num_pixels
        );

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(err));
    }

    return labels;
}

__global__ void naive_union_kernel(int* labels,
                            bool* changed,
                            int width,
                            int height){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;

    int idx = y * width + x;
    int this_label = labels[idx];  

    if(this_label == 0){
        return;
    }
    
    int new_label = this_label;

    // check neighbours 
    // north
    if(y > 0){
        int n_idx = idx - width;
        int north_label = labels[n_idx];
        if(north_label > 0 && north_label < new_label){
            new_label = north_label; 
        }
    }
    //south
    if(y < height - 1){
        int s_idx = idx + width;
        int south_label = labels[s_idx];
        if(south_label > 0 && south_label < new_label){
            new_label = south_label;
        }
    }
    // east
    if(x < width - 1){
        int e_idx = idx + 1;
        int east_label = labels[e_idx];
        if(east_label > 0 && east_label < new_label){
            new_label = east_label;
        }
    }
    // west
    if(x > 0){
        int w_idx = idx - 1; 
        int west_label = labels[w_idx];
        if(west_label > 0 && west_label < new_label){
            new_label = west_label; 
        }
    }

    if(new_label < this_label){
        atomicMin(&labels[idx], new_label);
        *changed = true;
    }
    
}

torch::Tensor naive_connected_components(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "Input must be uint8 (byte) binary image");

    auto height = input.size(0);
    auto width = input.size(1);
    
    auto labels = torch::zeros({height, width}, torch::dtype(torch::kInt32).device(input.device()));

    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);


    int num_pixels = width * height;
    thrust::transform(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_pixels),
        thrust::device_pointer_cast(labels.data_ptr<int>()),
        InitLabelsFunctor(input.data_ptr<unsigned char>())
    );

    auto changed_tensor = torch::zeros({1}, torch::dtype(torch::kBool).device(input.device()));
    bool* d_changed = changed_tensor.data_ptr<bool>();
    bool h_changed = true;

    int max_iter = width * height; 
    int iter = 0;

    while (h_changed && iter < max_iter) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        naive_union_kernel<<<blocks, threads>>>(
            labels.data_ptr<int>(),
            d_changed,
            width,
            height
        );

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;
    }

    return labels;
}