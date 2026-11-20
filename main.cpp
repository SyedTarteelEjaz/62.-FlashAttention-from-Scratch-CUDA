#include <torch/extension.h>

torch::Tensor flash_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_forward, "FlashAttention forward (online softmax, tiled, FP32)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"), pybind11::arg("causal") = false);
}
