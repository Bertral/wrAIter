py -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install whisper-mic==1.4.2 sentencepiece inquirerpy coqui-tts pygame pysbd accelerate diffusers
pip install --upgrade transformers==4.46.1
PAUSE