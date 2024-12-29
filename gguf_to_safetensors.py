import os
import argparse
import torch
import numpy as np
from safetensors.numpy import save_file
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
    """Convert a GGUF file to a safetensors file with dequantized data."""
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f"Extracted {len(tensors_metadata)} tensors from GGUF file")

    tensors_dict = {}

    for i, tensor_info in enumerate(tensors_metadata):
        tensor_name = tensor_info['name']
        shape = tensor_info['shape']
        quant_type_id = tensor_info['type']
        
        # 🔥 ここでタイプIDから量子化タイプ名を取得する
        quant_type = get_quant_type(quant_type_id)

        #print(f"Processing tensor: {tensor_name} | Shape: {shape} | Type: {quant_type}:{quant_type_id}")
        #ここ以下がコード修正対象（根本的に間違ってる）
        #Transformersの
        # for tensor in reader.tensors 
        #以下のdequantizeメソッド（llama.cppのggufパッケージ側の処理）を参考にする
        tensor_data = reader.get_tensor(i)
        name = tensor_data.name
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        # NumPy配列をPyTorchテンソルに変換する際に、非書き込み可能な配列をコピー
        weights = weights.copy()  # メモリの非書き込み制約を解除


        try:
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            weights_hf = torch.from_numpy(weights).to(dtype).numpy()
        except TypeError as e:
            print(f"TypeError occurred: {e}, fallback fp16")
            weights_hf = torch.from_numpy(weights.astype(np.float32)).to(torch.float16).numpy()

        print(f"dequantize tensor: {name} | Shape: {weights_hf.shape} | Type: {weights_hf.dtype}")

        tensors_dict[name] = weights_hf

    #GGUFからメタデータを取り出して付与する
    metadata = {"modelspec.architecture": f"{reader.get_field(Keys.General.FILE_TYPE)}", "description": "Model converted from gguf."}

    save_file(tensors_dict, output_path,metadata=metadata)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GGUF files to safetensors format.")
    parser.add_argument("--input", required=True, help="Path to the input GGUF file.")
    parser.add_argument("--output", required=True, help="Path to the output safetensors file.")
    parser.add_argument("--bf16", action="store_true", help="(onry cuda)Convert tensors to BF16 format instead of FP16.")

    args = parser.parse_args()

    convert_gguf_to_safetensors(args.input, args.output, args.bf16)
