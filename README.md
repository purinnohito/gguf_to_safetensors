This repository was created to deepen my personal understanding of GGUF files and because I wanted to convert models distributed in GGUF format, such as image generation models, back to safetensors.

Due to the rapid development pace of llama.cpp and the GGUF format, it appears that recent GGUF formats cannot be correctly converted to safetensors. As it is difficult to keep up with these changes, I have decided to end development at its current state.

The following repository may be useful for converting image generation models from GGUF to safetensors (as of May 13, 2025, I haven't tested it): https://github.com/calcuis/gguf


----

```
python gguf_to_safetensors.py --input mywait.gguf --output mywait.safetensors
```

----
Convert GGUF to safetensors

Specify the GGUF file to convert using --input and the desired safetensors filename using --output

Use --bf16 to save in BF16 precision (defaults to FP16 precision if not specified)

----
GGUFからsafetensorsへ変換する

変換したいGGUFファイルを--inputオプションで指定、
出力したいsafetensorsファイル名を--outputで指定


--bf16でsafetensorsファイルをBF16精度で保存(未指定時FP16精度)
