set -e
python3.12 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7
pip install whisper-mic==1.4.2 sentencepiece inquirerpy coqui-tts pygame pysbd diffusers accelerate
pip install --upgrade transformers==4.46.1
# quantization
# pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 torch==2.4.1+rocm6.1 torchaudio==2.4.1+rocm6.1 torchvision==0.19.1+rocm6.1
# pip install --force-reinstall --no-deps 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
read -sn 1 -p "Press any key to continue.."