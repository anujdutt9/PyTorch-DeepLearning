// One-stop header for TorchScript Library
#include <torch/script.h>
#include <iostream>
#include <memory>

// Main
int main(int argc, const char* argv[]) {
  // Take in TorchScript Model as Arg
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Initialize a TorchScript Module
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Module Loading Status
  std::cout << "ok\n";

// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
// Sample Input Image made of all ones with shape:
// (Batch_Size, Channels, Width, Height) => (1,3,224,224)
inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
// Print Out the Model Predictions/Outputs
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}