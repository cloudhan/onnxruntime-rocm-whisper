#################################################################################
# Convert Whisper model in ONNX from FP32 to FP16
#
# Run script as python3 scripts/fp32_to_fp16.py -s <size> -f <path to folder> -g
#################################################################################

import argparse
import os
import shutil

from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.transformers.fusion_options import FusionOptions

# Output format is (num_layers, num_heads, hidden_size)
MODEL_SIZE_INFO = {
    "tiny": (4, 6, 384),
    "base": (6, 8, 512),
    "small": (12, 12, 768),
    "medium": (24, 16, 1024),
    "large": (32, 20, 1280),
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s',
        '--size',
        required=False,
        default='tiny',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Size of Whisper model to load'
    )

    parser.add_argument(
        '-f',
        '--folder',
        required=False,
        type=str,
        default='./onnx/tiny',
        help="Root directory of the Whisper ONNX files",
    )

    parser.add_argument(
        '-g',
        '--gpu',
        action='store_true',
        help="Optimize for GPU",
    )
    parser.set_defaults(gpu=False)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    _, num_heads, hidden_size = MODEL_SIZE_INFO[args.size]
    optimization_options = FusionOptions("bart")
    use_external_data_format = args.size in {"small", "medium", "large"}

    fp16 = os.path.join(args.folder, "fp16")
    os.makedirs(fp16, exist_ok=True)
    fp32 = os.path.join(args.folder, "fp32")
    os.makedirs(fp32, exist_ok=True)

    # Generate FP16 and FP32 files
    files = os.listdir(args.folder)
    for fle in files:
        if ".onnx" in fle:
            file_path = os.path.join(args.folder, fle)
            
            optimization_options.use_multi_head_attention = "encoder" not in file_path
            m = optimize_model(
                file_path,
                model_type="bart",
                num_heads=num_heads,
                hidden_size=hidden_size,
                opt_level=0,
                optimization_options=optimization_options,
                use_gpu=args.gpu,
            )
            
            fp32_dest = file_path.replace(".onnx", "_fp32.onnx")
            m.save_model_to_file(fp32_dest, use_external_data_format)

            # m.convert_float_to_float16()
            m.convert_model_float32_to_float16(cast_input_output=False)

            fp16_dest = file_path.replace(".onnx", "_fp16.onnx")
            m.save_model_to_file(fp16_dest, use_external_data_format)
            
            os.remove(file_path)

    # Move files into FP16 and FP32 folders
    files = os.listdir(args.folder)
    for fle in files:
        file_path = os.path.join(args.folder, fle)
        fp16_path = os.path.join(args.folder, "fp16", fle.replace("_fp16.onnx", ".onnx"))
        fp32_path = os.path.join(args.folder, "fp32", fle.replace("_fp32.onnx", ".onnx"))

        if not os.path.isfile(file_path):
            continue

        if "fp16" in fle:
            os.rename(file_path, fp16_path)
        elif "fp32" in fle:
            os.rename(file_path, fp32_path)
        else:
            shutil.copyfile(file_path, fp16_path)
            shutil.copyfile(file_path, fp32_path)
            os.remove(file_path)


if __name__ == '__main__':
    main()
