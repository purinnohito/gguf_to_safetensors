import os
import argparse
import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
from typing import Dict, Tuple
from gguf import GGUFReader, dequantize
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, Keys

TYPE_TO_QUANT_MAP = {
    0: 'F32',
    1: 'F16',
    2: 'Q4_0',
    3: 'Q4_1',
    6: 'Q5_0',
    7: 'Q5_1',
    8: 'Q8_0',  # 🔥 ここが追加されたQ8タイプのマッピング
    9: 'Q8_1',
    10: 'Q2_K',
    11: 'Q3_K',
    12: 'Q4_K',
    13: 'Q5_K',
    14: 'Q6_K',
    15: 'Q8_K',
}

# タイプ番号から適切な量子化形式に変換する関数
def get_quant_type(type_id: int) -> str:
    """Type ID (整数) を量子化タイプ (文字列) にマッピングする関数。"""
    return TYPE_TO_QUANT_MAP.get(type_id, 'Unknown')


def load_gguf_and_extract_metadata(gguf_path: str) -> Tuple[GGUFReader, list]:
    """Load GGUF file and extract metadata and tensors."""
    reader = GGUFReader(gguf_path)
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {
            'name': tensor.name,
            'shape': tuple(tensor.shape.tolist()),
            'n_elements': tensor.n_elements,
            'n_bytes': tensor.n_bytes,
            'data_offset': tensor.data_offset,
            'type': tensor.tensor_type,
        }
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata


def convert_gguf_to_safetensors(gguf_path: str, output_path: str, use_bf16: bool) -> None:
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f"Extracted {len(tensors_metadata)} tensors from GGUF file")

    tensors_dict: dict[str, torch.Tensor] = {}

    for i, tensor_info in enumerate(tensors_metadata):
        tensor_name = tensor_info['name']
        shape = tensor_info['shape']
        quant_type_id = tensor_info['type']
        quant_type = get_quant_type(quant_type_id)

        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()

        try:
            # デバイスを確認し、適切なデータ型を設定
            if use_bf16:
                print(f"Attempting BF16 conversion")
                weights_tensor_tmp = torch.from_numpy(weights).to(dtype=torch.float32)
                weights_tensor = weights_tensor_tmp.clone().to(torch.bfloat16)
            else:
                print("Using FP16 conversion.")
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float16)

            weights_hf = weights_tensor
        except Exception as e:
            print(f"Error during BF16 conversion for tensor '{tensor_name}': {e}")
            weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(torch.float16)
            weights_hf = weights_tensor

        print(f"dequantize tensor: {tensor_name} | Shape: {weights_hf.shape} | Type: {weights_tensor.dtype}")

        tensors_dict[tensor_name] = weights_hf

    metadata = {"modelspec.architecture": f"{reader.get_field(Keys.General.FILE_TYPE)}", "description": "Model converted from gguf."}

    save_file(tensors_dict, output_path, metadata=metadata)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GGUF files to safetensors format.")
    parser.add_argument("--input", required=True, help="Path to the input GGUF file.")
    parser.add_argument("--output", required=True, help="Path to the output safetensors file.")
    parser.add_argument("--bf16", action="store_true", help="(onry cuda)Convert tensors to BF16 format instead of FP16.")

    args = parser.parse_args()

    convert_gguf_to_safetensors(args.input, args.output, args.bf16)
