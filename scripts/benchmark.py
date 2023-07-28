#########################################################
# Benchmark Whisper model
#
# See benchmark.sh for examples on how to run this file
#########################################################

import argparse
import datetime
import gc
import json
import librosa
import numpy as np
import os
import onnxruntime as ort
import psutil
import subprocess
import time
import torch
import torch.autograd.profiler as profiler
import whisper

from tqdm import trange
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import pipeline as pt_pipeline
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.pipelines import pipeline as ort_pipeline
from onnxruntime.transformers.benchmark_helper import measure_memory
from onnxruntime_extensions import get_library_path


# Output format is (num_layers, num_heads, hidden_size)
MODEL_SIZE_INFO = {
    "tiny": (4, 6, 384),
    "base": (6, 8, 512),
    "small": (12, 12, 768),
    "medium": (24, 16, 1024),
    "large": (32, 20, 1280),
    "large-v2": (32, 20, 1280),
}

PRECISION = {
    "fp32": (torch.float32, np.float32),
    "fp16": (torch.float16, np.float16),
    "int8": (torch.int8, np.int8),
}


# features_type: "" for none, "pt" for PyTorch, or "np" for NumPy
# load_with: "ffmpeg" for whisper.load_audio, "librosa" for librosa.load, "basic" for using AudioDecoder in ONNX model
def get_audio(args, load_with="ffmpeg", features_type=""):
    if features_type != "":
        assert load_with == "ffmpeg"

    def load_via_ffmpeg():
        audio = whisper.load_audio(args.audio_path)
        audio = whisper.pad_or_trim(audio)
        return audio

    def load_via_librosa():
        audio = librosa.load(args.audio_path)[0]
        audio = np.expand_dims(audio[: 30 * args.sample_rate], axis=0)
        return audio

    def load_via_basic():
        with open(args.audio_path, "rb") as fobj:
            audio = np.asarray(list(fobj.read()), dtype=np.uint8)
            audio = np.array([audio] * args.batch_size)
        return audio

    load_audio = (
        load_via_ffmpeg if load_with == "ffmpeg" else load_via_librosa if load_with == "librosa" else load_via_basic
    )

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    start_time = time.time()

    for _ in trange(args.num_runs):
        audio = load_audio()

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    end_time = time.time()

    print(f"Load audio: {(end_time - start_time) / args.num_runs} s")

    if features_type == "":
        return audio

    processor = AutoProcessor.from_pretrained(args.model_name)

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    start_time = time.time()

    for _ in range(args.num_runs):
        input_features = processor.feature_extractor(
            [audio] * args.batch_size, return_tensors=features_type
        ).input_features

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    end_time = time.time()

    print(f"Feature extraction: {(end_time - start_time) / args.num_runs} s")

    return input_features


def get_ort_inputs(args):
    if "encoder_model" in args.ort_model_path:
        # Encoder component of "Whisper export with optimum"
        ort_inputs = {
            "input_features": np.random.rand(args.batch_size, args.feature_size, args.encoder_seq_len).astype(
                np.float32
            ),
        }
        exclude_list = []
    elif "decoder_model" in args.ort_model_path:
        # Decoder component of "Whisper export with optimum"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, args.decoder_seq_len).astype(np.int64),
            "encoder_hidden_states": np.random.rand(
                args.batch_size, args.encoder_seq_len // 2, args.hidden_size
            ).astype(np.float32),
        }
        exclude_list = ["input_ids"]
    elif "decoder_with_past_model" in args.ort_model_path:
        # Decoder-with-past component of "Whisper export with optimum"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, 1).astype(np.int64),
        }
        for i in range(args.num_layers):
            past_kv = {
                f"past_key_values.{i}.decoder.key": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.decoder.value": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.encoder.key": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
                f"past_key_values.{i}.encoder.value": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
            }
            ort_inputs.update(past_kv)
        exclude_list = ["input_ids"]
    elif "_encoder_decoder_init" in args.ort_model_path:
        # Encoder-decoder-init component of "Whisper custom export with beam search"
        ort_inputs = {
            "encoder_input_ids": np.random.rand(args.batch_size, args.feature_size, args.encoder_seq_len).astype(
                np.float32
            ),
            "decoder_input_ids": np.random.rand(args.batch_size, 1).astype(np.int32),
        }
        exclude_list = ["decoder_input_ids"]
    elif "_decoder" in args.ort_model_path:
        # Decoder-with-past component of "Whisper custom export with beam search"
        ort_inputs = {
            "input_ids": np.random.rand(args.batch_size, 1).astype(np.int32),
        }
        for i in range(args.num_layers):
            past_kv = {
                f"past_key_self_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_value_self_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.past_decoder_seq_len, args.head_size
                ).astype(np.float32),
                f"past_key_cross_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
                f"past_value_cross_{i}": np.random.rand(
                    args.batch_size, args.num_heads, args.encoder_seq_len // 2, args.head_size
                ).astype(np.float32),
            }
            ort_inputs.update(past_kv)
        exclude_list = ["input_ids"]
    elif "beamsearch" in args.ort_model_path:
        # Whisper custom export with beam search contrib op
        input_features = get_audio(args, load_with="ffmpeg", features_type="np")
        ort_inputs = {
            "input_features": input_features,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            # "attention_mask": np.zeros((args.batch_size, args.feature_size, args.encoder_seq_len)).astype(np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    elif "all" in args.ort_model_path or "large-v2" in args.ort_model_path:
        # Whisper end-to-end ONNX model
        audio = get_audio(args, load_with="basic") if args.audio_path != "" else None
        ort_inputs = {
            "audio_stream": audio,
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
            # "attention_mask": np.zeros((args.batch_size, args.feature_size, args.encoder_seq_len)).astype(np.int32),
        }
        exclude_list = list(ort_inputs.keys())
    else:
        raise Exception("Unable to auto-detect inputs for provided model")

    return set_inputs(args, ort_inputs, exclude_list)


def get_hf_inputs(args, processor):
    if args.hf_api == "pipeline":
        # Only the audio is needed for inputs
        audio = get_audio(args, load_with="ffmpeg")
        hf_inputs = {"audio": audio}
        exclude_list = ["audio"]
    elif args.hf_api == "gen-and-dec":
        target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device
        input_features = get_audio(args, load_with="ffmpeg", features_type="pt")
        hf_inputs = {
            "inputs": input_features.to(target_device),
            "max_length": args.max_length,
            "min_length": args.min_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
            "use_cache": True,
        }
        exclude_list = [key for key in hf_inputs if key not in {"inputs"}]
    else:
        raise Exception("Could not calculate model inputs")

    return set_inputs(args, hf_inputs, exclude_list)


def set_inputs(args, input_dict, exclude_list):
    # Cast certain inputs to another dtype
    for k, v in input_dict.items():
        if k in exclude_list:
            continue

        if isinstance(v, torch.Tensor):
            input_dict[k] = v.to(PRECISION[args.precision][0])
        elif isinstance(v, np.ndarray):
            input_dict[k] = v.astype(PRECISION[args.precision][1])

    return input_dict


def get_vars(args):
    inputs, processor, model, pipeline = None, None, None, None
    if args.benchmark_type in {"HF + PT", "HF + PT2"}:
        processor, model, pipeline = get_hf_pt(args)
    elif args.benchmark_type == "HF + ORT":
        processor, model, pipeline = get_hf_ort(args)
    elif args.benchmark_type == "ORT":
        model = get_ort_model(args)
    else:
        raise Exception("Invalid benchmark type provided")

    # Get inputs
    if args.benchmark_type == "ORT":
        inputs = get_ort_inputs(args)
        if (
            "audio_stream" not in set(map(lambda model_input: model_input.name, model.get_inputs()))
            and "audio_stream" in inputs
        ):
            # Remove when 'audio' input is not in model
            del inputs["audio_stream"]
    else:
        inputs = get_hf_inputs(args, processor)
        if args.hf_api == "pipeline":
            inputs = inputs["audio"]

    return inputs, processor, model, pipeline


def get_hf_pt(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    torch_dtype = PRECISION[args.precision][0] if args.precision != "int8" else PRECISION["fp32"][0]
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device

    if args.hf_pt_model_path == "":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
        ).to(target_device)
    else:
        assert os.path.exists(args.hf_pt_model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.hf_pt_model_path,
            torch_dtype=torch_dtype,
        ).to(target_device)

    if "PT2" in args.benchmark_type:
        model = torch.compile(model)

    pipeline = pt_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=(-1 if args.device == "cpu" else args.device_id),
        return_timestamps=True,
        chunk_length_s=(30 if args.long_audio else 0),
    )
    return processor, model, pipeline


def get_hf_ort(args):
    processor = AutoProcessor.from_pretrained(args.model_name)
    torch_dtype = PRECISION[args.precision][0]
    target_device = f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device
    use_io_binding = args.device == "cuda"

    if args.hf_ort_model_path == "":
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            from_transformers=True,
            use_io_binding=use_io_binding,
        ).to(target_device)

        if args.precision == "fp16":
            # Save model
            args.hf_ort_model_path = f"hf-ort-optimum-whisper-{args.model_size}"
            model.save_pretrained(args.hf_ort_model_path)

            # Fuse model and convert to FP16
            subprocess.run(
                [
                    "python3",
                    "whisper/scripts/fp32_to_fp16.py",
                    "--size",
                    args.model_size,
                    "--folder",
                    args.hf_ort_model_path,
                    "--gpu",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Reload fused FP16 model
            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                os.path.join(args.hf_ort_model_path, "fp16"),
                use_io_binding=use_io_binding,
            ).to(target_device)

        elif args.precision == "int8":
            raise NotImplementedError("Script to automate INT8 quantization for HF-ORT is not implemented")

    else:
        assert os.path.exists(args.hf_ort_model_path)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.hf_ort_model_path,
            use_io_binding=use_io_binding,
        ).to(target_device)

    pipeline = ort_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=(-1 if args.device == "cpu" else args.device_id),
        return_timestamps=True,
        chunk_length_s=(30 if args.long_audio else 0),
    )
    return processor, model, pipeline


def get_ort_model(args):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = args.profile
    sess_options.register_custom_ops_library(get_library_path())
    if args.verbose:
        sess_options.log_verbosity_level = 3
        sess_options.log_severity_level = 1

    provider_options = {}
    if args.execution_provider in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
        provider_options["device_id"] = args.device_id
    if args.tuning:
        provider_options["tunable_op_enable"] = 1
        provider_options["tunable_op_tuning_enable"] = 1
        provider_options["tunable_op_max_tuning_duration_ms"] = 500

    sess = ort.InferenceSession(
        args.ort_model_path, sess_options, providers=[(args.execution_provider, provider_options)]
    )

    return sess


# Benchmark types: HF + PT, HF + PT2, HF + ORT
def run_hf_pipeline_inference(args, audio, pipe):
    if args.profile:
        # Profile kernels
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            outputs = pipe([audio] * args.batch_size)
            # Filename format example: "hf_pt2_pipeline_<current-time>.txt"
            filename = f"{args.benchmark_type.lower().replace(' + ', '_')}_pipeline_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
            fobj = open(filename, "w")
            fobj.write(
                prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)
            )
            fobj.close()

        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        outputs = pipe([audio] * args.batch_size)
        print(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: pipe([audio] * args.batch_size))

        return

    # Warm up
    outputs = None
    for _ in range(args.warmup_runs):
        outputs = pipe([audio] * args.batch_size)

    if args.verbose:
        print("=" * 64 + " output " + "=" * 64)
        print(outputs)

    # Benchmark
    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    start_time = time.time()

    for _ in trange(args.num_runs):
        pipe([audio] * args.batch_size)

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} qps")


# Benchmark types: HF + PT, HF + PT2, HF + ORT
def run_hf_generate_and_decode_inference(args, inputs, processor, model):
    def gen_and_dec():
        # HF + PT, HF + PT2, HF + ORT
        # HF + ORT works with latest optimum version
        predicted_ids = model.generate(**inputs)
        transcription = []
        for bs in range(args.batch_size):
            for rs in range(args.num_return_sequences):
                transcription.append(
                    processor.batch_decode(
                        predicted_ids[bs * args.num_return_sequences + rs], skip_special_tokens=True
                    )[0]
                )
        return transcription

    if args.profile:
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            gen_and_dec()
            # Filename format example: "hf_pt2_gen_and_dec_<current-time>.txt"
            filename = f"{args.benchmark_type.lower().replace(' + ', '_')}_gen_and_dec_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
            fobj = open(filename, "w")
            fobj.write(
                prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)
            )
            fobj.close()

        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        gen_and_dec()
        print(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: gen_and_dec())

        return

    # Warm up
    transcription = None
    for _ in range(args.warmup_runs):
        transcription = gen_and_dec()

    if args.verbose:
        print("=" * 64 + " output " + "=" * 64)
        print(transcription)

    # Benchmark
    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    start_time = time.time()

    for _ in trange(args.num_runs):
        gen_and_dec()

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} qps")


# Benchmark types: ORT only
def run_ort_only_inference(args, inputs, model):
    if args.profile:
        # Measure CPU usage
        pid = os.getpid()
        process = psutil.Process(pid)
        process.cpu_percent(interval=0.1)

        model.run(None, inputs)
        print(f"CPU percentage = {process.cpu_percent(interval=None)}%")

        # Get model with profiling off to stop generating logs
        args.profile = False
        model = get_ort_model(args)

        # Measure memory usage
        gc.collect()
        torch.cuda.empty_cache()
        measure_memory(is_gpu=(args.device == "cuda"), func=lambda: model.run(None, inputs))

        return

    # Warm up
    outputs = None
    for _ in range(args.warmup_runs):
        outputs = model.run(None, inputs)[0]

    if args.verbose:
        print("=" * 64 + " output " + "=" * 64)
        tok = AutoProcessor.from_pretrained(f"openai/whisper-{args.model_size}").tokenizer
        for o in outputs.reshape((-1,) + outputs.shape[-1:]):
            print(tok.convert_tokens_to_string(tok.convert_ids_to_tokens(o)))

    # Benchmark
    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    start_time = time.time()

    for _ in range(args.num_runs):
        model.run(None, inputs)

    if args.device == "cuda":
        torch.cuda.synchronize()
        print("Device synced at time", time.time())
    end_time = time.time()

    latency = (end_time - start_time) / args.num_runs
    print(f"Batch size = {args.batch_size}, latency = {latency} s, throughput = {args.batch_size / latency} qps")


def run_inference(args, inputs, processor, model, pipeline):
    if args.hf_api == "pipeline":
        run_hf_pipeline_inference(args, inputs, pipeline)
    elif args.hf_api == "gen-and-dec":
        run_hf_generate_and_decode_inference(args, inputs, processor, model)
    else:
        run_ort_only_inference(args, inputs, model)


def parse_args():
    parser = argparse.ArgumentParser()

    # Args for benchmark type
    parser.add_argument(
        "-bt", "--benchmark-type", type=str, required=True, choices=["HF + PT", "HF + PT2", "HF + ORT", "ORT"]
    )
    parser.add_argument(
        "--hf-api",
        type=str,
        choices=["pipeline", "gen-and-dec"],
        help="Whether to use Hugging Face's 'pipeline()' API or \
                        'model.generate() + processor.batch_decode()' API",
    )

    # Args for audio file and batch size
    parser.add_argument("-a", "--audio-path", type=str, default="", help="Path to audio file for E2E evaluation")
    parser.add_argument(
        "--long-audio", default=False, action="store_true", help="Whether the audio file is longer than 30s"
    )
    parser.add_argument("-b", "--batch-size", required=True, type=int, default=1)

    # Args for choosing the model
    parser.add_argument(
        "-s",
        "--model-size",
        required=True,
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
    )
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int8", "fp16", "fp32"],
        help="Precision for model and inputs. For PyTorch models, this sets the model's precision. \
                              For ONNX models, the model's precision should be set before running this script. ",
    )
    parser.add_argument(
        "--hf-pt-model-path",
        type=str,
        default="",
        help="Path to directory containing all PyTorch files (e.g. tokenizer, PyTorch model)",
    )
    parser.add_argument(
        "--hf-ort-model-path",
        type=str,
        default="",
        help="Path to directory containing all ONNX files (e.g. tokenizer, encoder, decoder, decoder_with_past)",
    )
    parser.add_argument("--ort-model-path", type=str, default="", help="Path to ONNX model")

    ######################################################################################
    # Args for ORT-only benchmarking

    # Args for ORT E2E and beam search decoding
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "-f", "--feature-size", type=int, default=80, help="Known as 'feature size' or 'number of Mels'"
    )
    parser.add_argument(
        "-es",
        "--encoder-seq-len",
        type=int,
        default=3000,
        help="Known as 'encoder sequence length' or 'number of frames'",
    )

    # When skipping pre/post processing and not evaluating E2E (e.g. ORT component-wise),
    # the following input args can be used:

    # Dynamic inputs:
    parser.add_argument(
        "-ds", "--decoder-seq-len", type=int, default=448, help="Maximum decoder sequence length is 448"
    )
    parser.add_argument(
        "-pds", "--past-decoder-seq-len", type=int, default=447, help="Maximum past decoder sequence length is 447"
    )
    ######################################################################################

    # Args for running and evaluating the model
    parser.add_argument(
        "-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda", "rocm"]
    )
    parser.add_argument("-id", "--device-id", type=int, default=0, help="GPU device ID when using CUDA")
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-r", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Args for accessing detailed info
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Whether to profile the model (e.g. CPU usage, memory footprint)",
    )
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print information (e.g. outputs, verifications)",
    )

    # Args for offline tuning
    parser.add_argument(
        "-t",
        "--tuning",
        action="store_true",
        help="Enable TunableOp and tuning. "
        "This will incur longer warmup latency, and is mandatory for some operators of ROCm EP.",
    )

    parser.add_argument(
        "-tl",
        "--tuning_results_load_path",
        required=False,
        default=None,
        type=str,
        help="Load tuning results from the path for the InferenceSessions.",
    )

    parser.add_argument(
        "-ts",
        "--tuning_results_save_path",
        required=False,
        default=None,
        type=str,
        help="Merge and save tuning results to the path from the InferenceSessions.",
    )

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")
    if args.device == "rocm":
        # pytorch expect device to be cuda for both cuda and rocm
        setattr(args, "device", "cuda")

    if args.benchmark_type == "ORT":
        args.hf_api = None

    # Set model properties
    (num_layers, num_heads, hidden_size) = MODEL_SIZE_INFO[args.model_size]
    setattr(args, "model_name", f"openai/whisper-{args.model_size}")
    setattr(args, "num_layers", num_layers)
    setattr(args, "num_heads", num_heads)
    setattr(args, "hidden_size", hidden_size)
    setattr(args, "head_size", hidden_size // num_heads)

    return args


def recursive_visit(obj, func, visited=None):
    def recursive_visit_impl(visited, obj, func):
        if id(obj) in visited:
            return

        visited.add(id(obj))
        func(obj)

        if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
            return

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                func(v)
                recursive_visit_impl(visited, v, func)
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                func(k)
                func(v)
                recursive_visit_impl(visited, k, func)
                recursive_visit_impl(visited, v, func)
            return

        if getattr(obj, "__dict__", None) is not None:
            for k, v in vars(obj).items():
                func(k)
                func(v)
                recursive_visit_impl(visited, k, func)
                recursive_visit_impl(visited, v, func)
            return

        if getattr(obj, "__iter__", None) is not None:
            for v in obj:
                func(v)
                recursive_visit_impl(visited, v, func)
            return

        print("    >>>> recursive_visit cannot process", type(obj))

    if visited is None:
        visited = set()
    visited.add(id(visited))
    visited.add(id(func))
    recursive_visit_impl(visited, obj, func)


def get_all_inference_sessions(args, model, pipeline):
    sessions = {}
    if args.tuning_results_load_path or args.tuning_results_save_path:

        def get_all_sessions(obj):
            if isinstance(obj, ort.InferenceSession):
                sessions[id(obj)] = obj

        recursive_visit(model, get_all_sessions)
        recursive_visit(pipeline, get_all_sessions)

    return list(sessions.values())


def load_tuning_results(args, sessions):
    if args.tuning_results_load_path is None or len(sessions) == 0:
        return

    if not os.path.exists(args.tuning_results_load_path):
        print(f"ERROR: Cannot find tuning results {args.tuning_results_load_path}")
        return

    trs = json.load(open(args.tuning_results_load_path))
    for sess in sessions:
        sess.set_tuning_results(trs)


def save_tuning_results(args, sessions):
    if args.tuning_results_load_path is None or len(sessions) == 0:
        return

    from onnxruntime.tools import offline_tuning

    m = offline_tuning.Merger()
    for sess in sessions:
        m.merge(sess.get_tuning_results())
    trs = m.get_merged()
    json.dump(trs, open(args.tuning_results_save_path, "w"))


def main():
    args = parse_args()
    print(args.__dict__)
    if args.tuning:
        # NOTE: hf pipeline, these envvars are for developers only, and should not be used in production env
        os.environ["ORT_CUDA_TUNABLE_OP_ENABLE"] = "1"
        os.environ["ORT_CUDA_TUNABLE_OP_TUNING_ENABLE"] = "1"
        os.environ["ORT_ROCM_TUNABLE_OP_ENABLE"] = "1"
        os.environ["ORT_ROCM_TUNABLE_OP_TUNING_ENABLE"] = "1"

    torch.backends.cudnn.benchmark = True
    inputs, processor, model, pipeline = get_vars(args)
    sessions = get_all_inference_sessions(args, model, pipeline)
    load_tuning_results(args, sessions)
    run_inference(args, inputs, processor, model, pipeline)
    save_tuning_results(args, sessions)


if __name__ == "__main__":
    main()
