#include <torch/extension.h>


__global__ void d_relu(float *a, float *res, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        if (*(a+i) > 0.0) {
            *(res+i) = *(a+i);
        }
        else {
            *(res+i) = 0;
        }
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int block_size = 128;


__forceinline__ int calc_grid_size(int m) {
    return (m + block_size - 1) / block_size;
}


torch::Tensor relu(torch::Tensor a) {
    CHECK_INPUT(a);

    auto res = torch::empty_like(a);
    int n = a.numel();

    d_relu<<<calc_grid_size(n), block_size>>>(
        a.data_ptr<float>(),
        res.data_ptr<float>(),
        n
    );

    return res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_relu", &relu, "Custom vector ReLU-function");
}
