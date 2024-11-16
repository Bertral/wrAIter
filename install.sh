set -e
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install whisper-mic==1.4.2 sentencepiece inquirerpy coqui-tts pygame pysbd accelerate diffusers gguf
pip install --upgrade transformers==4.46.1
read -sn 1 -p "Press any key to continue.."
