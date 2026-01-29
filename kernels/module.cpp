#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img);
torch::Tensor box_blur(torch::Tensor img, int blurSize);
torch::Tensor sobel(torch::Tensor img);
torch::Tensor dilation(torch::Tensor img, int filterSize);
torch::Tensor erosion(torch::Tensor img, int filterSize);
torch::Tensor connected_components(torch::Tensor input);
torch::Tensor naive_connected_components(torch::Tensor input);
torch::Tensor connected_components_shared_no_loop(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
    m.def("blur", &box_blur, "box blur kernel");
    m.def("sobel", &sobel, "sobel filter kernel");
    m.def("dilate", &dilation, "dilation kernel");
    m.def("erode", &erosion, "erosion kernel");
    m.def("connectedComponents", &connected_components, "Connected Component Labeling kernel");
    m.def("naiveConnectedComponents", &naive_connected_components, "naive Connected Component Labeling");
}