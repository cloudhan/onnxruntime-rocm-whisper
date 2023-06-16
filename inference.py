# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# based on Peter Mcaughan's script

from onnxruntime import InferenceSession, SessionOptions

import argparse
import gc
import numpy as np
import os
import onnxruntime
import psutil
import time
import logging
import sys
import torch

BEAM_SIZES = [4]
BATCH_SIZES = [1]
# BEAM_SIZES = [1,2,4]
# BATCH_SIZES = [1,2,3,4,5,6]
NUM_RETURN_SEQUENCES = 1
MAX_ITER = 1


def prepare_data(args):
    from transformers import AutoProcessor
    from datasets import load_dataset

    processor = AutoProcessor.from_pretrained(args.model_dir)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    return [input_features, processor]


def run_inference(args, input_features, ort_ep):
    """Run inference with Whisper Pytorch and ONNX models.

    Run an easy example with two models for a simple check on performance.

    Args:
        args: User input.
    """

    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = args.repetition_penalty

    sess_options = SessionOptions()
    sess_options.log_severity_level = 4
    sess = InferenceSession(args.onnx_path, sess_options, providers=[ort_ep])
    warmup_run = False

    ort_time_cost_batch_beam = {}
    ort_out_batch_beam = {}
    for bsz in BATCH_SIZES:
        input_data = input_features.repeat(bsz, 1, 1)
        for beam_size in BEAM_SIZES:
            print("~~~~~~~ \t BATCH_SIZE ", bsz, "\t BEAM_SIZE: ", beam_size, " \t~~~~~~~")
            ort_inputs = {
                "input_features": input_data.cpu().numpy().astype(np.float32),
                "max_length": np.array([max_length], dtype=np.int32),
                "min_length": np.array([min_length], dtype=np.int32),
                "num_beams": np.array([beam_size], dtype=np.int32),
                "num_return_sequences": np.array([NUM_RETURN_SEQUENCES], dtype=np.int32),
                "length_penalty": np.array([1.0], dtype=np.float32),
                "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
            }

            # Warmup run if very first
            if warmup_run:
                out = sess.run(None, ort_inputs)
                warmup_run = False
            # Timed run
            print("After warmup run")
            ort_out = {}
            start_time = time.time()
            for iter in range(MAX_ITER):
                ort_out = sess.run(None, ort_inputs)
            ort_time_cost = (time.time() - start_time) / MAX_ITER
            print("\t ORT ", ort_ep, ": ", ort_time_cost, "s")

            ort_time_cost_batch_beam[bsz * len(BEAM_SIZES) + beam_size] = ort_time_cost
            ort_out_batch_beam[bsz * len(BEAM_SIZES) + beam_size] = ort_out

    return [ort_time_cost_batch_beam, ort_out_batch_beam]


def run_inference_pt(args, input_features, processor, ort_time_cost_batch_beam, ort_out_batch_beam, is_cuda):
    """Run inference with Whisper Pytorch and ONNX models.

    Run an easy example with two models for a simple check on performance.

    Args:
        args: User input.
    """
    import torch

    if is_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer

    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = torch.tensor(args.repetition_penalty).half()
    no_repeat_ngram_size = args.no_repeat_ngram_size
    tokenizer = WhisperTokenizer.from_pretrained(args.model_dir)

    with torch.no_grad():
        model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float16)
        if is_cuda:
            model.to(device)

        for bsz in BATCH_SIZES:
            input_data = input_features.repeat(bsz, 1, 1).half()
            if is_cuda:
                input_data = input_data.to(device)
                input_data.is_cuda
            for beam_size in BEAM_SIZES:
                print("~~~~~~~ \t BATCH_SIZE ", bsz, "\t BEAM_SIZE: ", beam_size, " \t~~~~~~~")
                py_out = {}
                start_time = time.time()
                for iter in range(MAX_ITER):
                    py_out = model.generate(
                        input_data,
                        decoder_start_token_id=tokenizer.bos_token_id,
                        num_beams=beam_size,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        min_length=min_length,
                        max_length=max_length,
                        length_penalty=torch.tensor(1.0).half(),  ##1.0,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        attention_mask=torch.ones(input_features.shape),
                        early_stopping=True,
                        use_cache=True,
                    )
                    # py_out = model.generate(input_data, num_beams=beam_size, num_return_sequences=NUM_RETURN_SEQUENCES,)

                py_time_cost = (time.time() - start_time) / MAX_ITER
                print("\t PyTorch ", device, ": ", py_time_cost, "s")

                print(
                    "\t Gain over PyTorch: ",
                    100.0
                    * (ort_time_cost_batch_beam[bsz * len(BEAM_SIZES) + beam_size] - py_time_cost)
                    / (py_time_cost),
                )

                # Test Parity
                py_decoded = []
                py_decoded_out = processor.batch_decode(py_out, skip_special_tokens=True)
                for j in range(bsz):
                    for i in range(NUM_RETURN_SEQUENCES):
                        py_decoded.append(py_decoded_out[j * NUM_RETURN_SEQUENCES + i])  ##.cpu()
                        # print(f"pytorch index {j * NUM_RETURN_SEQUENCES + i} : {py_decoded_out[j * NUM_RETURN_SEQUENCES + i]}")

                ort_decoded = []
                ort_out = ort_out_batch_beam[bsz * len(BEAM_SIZES) + beam_size]
                for j in range(bsz):
                    ort_decoded_output = processor.batch_decode(
                        torch.from_numpy(ort_out[0][j]), skip_special_tokens=True
                    )
                    for i in range(NUM_RETURN_SEQUENCES):
                        ort_decoded.append(ort_decoded_output[i])
                        # print(f"ort index {j * NUM_RETURN_SEQUENCES + i} : {ort_decoded_output[i]}")

                unequal = 0
                for i in range(len(ort_decoded)):
                    print("~~~~~~~~~~~~~~~Difference:~~~~~~~~~~~~~~~")
                    print(f"index {i} : ort : {ort_decoded[i]}")
                    print(f"index {i} : torch : {py_decoded[i]}")
                #     if ort_decoded[i] != py_decoded[i]:
                #         unequal += 1
                #         print("~~~~~~~~~~~~~~~Difference:~~~~~~~~~~~~~~~")
                #         print("\t \t \t ORT:", ort_decoded[i])
                #         print("\t \t \t PY:", py_decoded[i])
                # if (unequal == 0):
                #     print("\t Parity verified.")
                # else:
                #     raise Exception("Parity issue found with bsz ", bsz, " and beam_size ", beam_size)


# GLOBAL ENVS
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate")


def print_args(args):
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


def user_command():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--max_length", type=int, default=20, help="default to 20")
    parent_parser.add_argument("--min_length", type=int, default=0, help="default to 0")
    parent_parser.add_argument("-i", "--input", type=str, default="./", help="default to current dir.")
    parent_parser.add_argument("-b", "--num_beams", type=int, default=5, help="default to 5")
    parent_parser.add_argument("-bsz", "--batch_size", type=int, default=1, help="default to 1")
    parent_parser.add_argument("--repetition_penalty", type=float, default=1.0, help="default to 1.0")
    parent_parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="default to 3")

    required_args = parent_parser.add_argument_group("required input arguments")
    required_args.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=False,
        default="openai/whisper-large",
        help="The OpenAi whisper model for PyTorch comparison \
                               An official model looks like openai/whisper-large.",
    )

    required_args.add_argument("--onnx_path", type=str)

    print_args(parent_parser.parse_args())
    return parent_parser.parse_args()


if __name__ == "__main__":
    args = user_command()
    [input_features, processor] = prepare_data(args)
    [ort_time_cost_batch_beam, ort_out_batch_beam] = run_inference(args, input_features, "ROCMExecutionProvider")
    run_inference_pt(args, input_features, processor, ort_time_cost_batch_beam, ort_out_batch_beam, True)
