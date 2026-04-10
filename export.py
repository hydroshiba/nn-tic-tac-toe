import sys
import torch
import random
from tqdm import tqdm
from component import board, architecture, agent

def export_to_onnx(model, filename):
	model.eval()
	dummy_input = torch.zeros(1, 9)
	torch.onnx.export(model, dummy_input, filename,
        input_names=["input"], output_names=["policy", "value"],
        opset_version=18,
        external_data=False
    )

def get_architecture(state_dict):
	# Adapt older state_dicts that used 'policy' and 'value' instead of 'policy_head' and 'value_head'
	adapted_dict = {}
	for k, v in state_dict.items():
		k = k.replace("policy.", "policy_head.")
		k = k.replace("value.", "value_head.")
		adapted_dict[k] = v

	for name in dir(architecture):
		cls = getattr(architecture, name)
		if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
			try:
				model = cls()
				model.load_state_dict(adapted_dict)
				return model
			except Exception:
				continue
	return None

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python export.py <path>")
		sys.exit(1)
	
	path = sys.argv[1] 
	state_dict = torch.load(path, map_location=torch.device('cpu'))
	model = get_architecture(state_dict)
	
	if model is None:
		print("Could not find matching architecture.")
		sys.exit(1)
	
	out_filename = path.replace(".pth", ".onnx")
	export_to_onnx(model, out_filename)
	print(f"Exported to {out_filename}") 