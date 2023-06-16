# build ORT

```bash
# NOTE the branch is extensively annotated with debug log and may not suitable for profiling, checkout previous commit
git clone https://github.com/microsoft/onnxruntime.git -b guangyunhan/decoding-for-amd-e2e $HOME/onnxruntime
cd $HOME/onnxruntime
./build_rocm.sh  # build to $HOME/build_rocm/Release
```

# export whisper

```bash
pip install protobuf==3.20.2 # DONT USE protobuf 4.*, otherwise, segfault
export PYTHONPATH=$HOME/onnxruntime/build_rocm/Release/build/lib

# swap `whisper-large` to `whisper-small` or `whisper-tiny` for faster export and smaller model for testing
export W=whisper-large

python -m onnxruntime.transformers.models.whisper.convert_to_onnx \
    -m openai/$W \
    --output $W \
    --use_gpu \
    --provider rocm \
    --precision fp16 \
    --optimize_onnx \
    --use_external_data_format

export ORT_ROCM_TUNABLE_OP_TUNING_ENABLE=1
export ORT_ROCM_TUNABLE_OP_ENABLE=1
python inference.py \
    --min_length 7 \
    --max_length 128 \
    --repetition_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    -m openai/$W \
    --onnx_path ./$W/openai/${W}_beamsearch.onnx

# You should see the decoded sequence with ort: Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.
# You can tweak BEAM_SIZES and BATCH_SIZES in inference.py
```
