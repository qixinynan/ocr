# Local GPU OCR

This deployment targets NVIDIA ARM64 machines and uses an open-source OCR model stack.

Default runtime:

- Base image: `nvcr.io/nvidia/vllm:26.05.post1-py3`
- OCR engine: EasyOCR with CUDA when available
- Default languages: Simplified Chinese + English
- API port: `8008`
- Model cache: `./models/easyocr`

On this DGX Spark / GB10 setup, EasyOCR's CRAFT detector produced bad boxes
through cuDNN in the NVIDIA vLLM 26.05 image. The service disables cuDNN while
keeping CUDA enabled.

## Run

```bash
sudo docker compose up -d --build
```

## Check

```bash
curl http://127.0.0.1:8008/health
curl -F file=@/path/to/image.png http://127.0.0.1:8008/ocr
```

Override languages:

```bash
curl -F file=@/path/to/image.png -F langs=ch_sim,en http://127.0.0.1:8008/ocr
```

Models are cached under `./models`.

## Status

Verified on this machine:

- Container: `local-gpu-ocr`
- GPU device reported by OCR: `cuda`
- GPU name: `NVIDIA GB10`
- Health endpoint: `http://127.0.0.1:8008/health`
