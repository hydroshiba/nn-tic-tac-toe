import os
import json

def generate_index():
    model_dir = "model"
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} not found.")
        return

    # Find all .onnx files
    onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
    onnx_files.sort()

    index_path = os.path.join(model_dir, "models.json")
    with open(index_path, "w") as f:
        json.dump(onnx_files, f, indent=2)
    
    print(f"Successfully generated {index_path} tracking {len(onnx_files)} models:")
    for m in onnx_files:
        print(f" - {m}")

if __name__ == "__main__":
    generate_index()