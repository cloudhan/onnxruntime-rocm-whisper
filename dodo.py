def task_get_sample():
    return {
        "actions": [
            "mkdir -p ./data",
            "wget https://github.com/microsoft/onnxruntime-extensions/raw/01d3905801f25bb0a159c4de6a5ae5120ca5c6a0/test/data/1272-141231-0002.mp3 -nv -O ./data/1272-141231-0002.mp3",
            "wget https://github.com/openai/whisper/raw/b91c907694f96a3fb9da03d4bbdc83fbcd3a40a4/tests/jfk.flac -nv -O ./data/jfk.flac",
        ],
        "file_dep": [
            "dodo.py",
        ],
        "targets": [
            "./data/1272-141231-0002.mp3",
            "./data/jfk.flac",
        ],
    }

def task_get_onnx_model():
    return {
        "actions": [
            "mkdir -p ./pipelines/tmp/hf-ort-optimum-whisper/%(size)s",
            "python scripts/hf_to_onnx.py -s %(size)s  --path ./pipelines/tmp/hf-ort-optimum-whisper",
            "cd ./pipelines/tmp/hf-ort-optimum-whisper/%(size)s && python ../../../../scripts/fp32_to_fp16.py --gpu -s %(size)s --folder ./",
            "mv ./pipelines/tmp/hf-ort-optimum-whisper/%(size)s ./pipelines/hf-ort-optimum-whisper-%(size)s"
        ],
        "params": [
            {"name": "size", "long": "size", "type": str, "default": "large", "choices": [("tiny", ""), ("small", ""), ("large", "")]},
        ],
        "verbosity": 2
    }

# TODO: add ORT E2E (pre/post processing + beam search op)


def task_benchmark():
    prefix = "python ./scripts/benchmark.py --verbose --audio-path data/%(audio)s -p %(precision)s -s %(size)s -b %(batch)s -r %(iter)s -d %(dev)s "
    tuning_options = " -t -tl tuning_results.json -ts tuning_results.json"
    return {
        "actions": [
            # prefix + '-bt "HF + PT"  --hf-api pipeline',
            # prefix + '-bt "HF + PT"  --hf-api gen-and-dec',
            # prefix + '-bt "HF + PT2" --hf-api pipeline',
            # prefix + '-bt "HF + PT2" --hf-api gen-and-dec',
            # prefix + '-bt "HF + ORT" --hf-api pipeline    --hf-ort-model-path pipelines/hf-ort-optimum-whisper-%(size)s/%(precision)s ' + tuning_options,
            # prefix + '-bt "HF + ORT" --hf-api gen-and-dec --hf-ort-model-path pipelines/hf-ort-optimum-whisper-%(size)s/%(precision)s ' + tuning_options,
            prefix + '-bt ORT --ort-model-path whisper-%(size)s/openai/whisper-%(size)s_beamsearch.onnx --max-length 256 ' + tuning_options,
        ],
        "params": [
            {"name": "batch", "long": "batch", "type": int, "default": 2},
            {"name": "iter", "long": "iter", "type": int, "default": 20},
            {"name": "dev", "long": "dev", "type": str, "default": "rocm", "choices": [("cpu", ""), ("cuda", ""), ("rocm", "")]},
            {"name": "size", "long": "size", "type": str, "default": "large", "choices": [("tiny", ""), ("large", "")]},
            {"name": "precision", "long": "precision", "type": str, "default": "fp16", "choices": [("fp16", ""), ("fp32", "")]},
            {"name": "audio", "long": "audio", "type": str, "default": "1272-141231-0002.mp3", "choices": [("1272-141231-0002.mp3", ""), ("jfk.flac", "")]},
        ],
        "verbosity": 2
    }
