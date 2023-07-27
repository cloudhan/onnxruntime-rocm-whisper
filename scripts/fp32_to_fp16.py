#################################################################################
# Convert Whisper model in ONNX from FP32 to FP16
#
# Run script as python3 scripts/fp32_to_fp16.py -s <size> -f <path to folder> -g
#################################################################################

import argparse
import os
import shutil
import copy

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
    optimization_options.disable_multi_head_attention_bias = True

    use_external_data_format = args.size in {"tiny", "small", "medium", "large"}

    fp16_dir = os.path.join(args.folder, "fp16")
    fp32_dir = os.path.join(args.folder, "fp32")
    os.makedirs(fp16_dir, exist_ok=True)
    os.makedirs(fp32_dir, exist_ok=True)

    # Generate FP16 and FP32 files
    files = os.listdir(args.folder)
    for fle in files:
        if fle.endswith(".onnx"):
            print("Processing", fle)
            if fle.endswith("_fp32.onnx") or fle.endswith("_fp16.onnx"):
                print("Skipped")
                continue
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

            copy.deepcopy(m).save_model_to_file(os.path.join(fp32_dir, fle), use_external_data_format)

            # m.convert_float_to_float16()
            m.convert_float_to_float16(cast_input_output=False)
            m.save_model_to_file(os.path.join(fp16_dir, fle), use_external_data_format)

            os.remove(file_path)
            print("Done")

    # Move files into FP16 and FP32 folders
    files = os.listdir(args.folder)
    for fle in files:
        file_path = os.path.join(args.folder, fle)

        if not os.path.isfile(file_path) or fle.endswith(".onnx"):
            continue

        if fle.endswith(".onnx_data"):
            os.remove(file_path)
            continue

        shutil.copy2(file_path, fp32_dir)
        shutil.copy2(file_path, fp16_dir)
        os.remove(file_path)


if __name__ == '__main__':
    main()
