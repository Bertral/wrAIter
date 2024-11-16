import json
import os.path


class Settings:
    def __init__(self, file='settings.json'):
        self.file = file
        self.defaults = {
            'silent': False,  # mute audio output
            'censor': True,  # censor bad words (slurs/nsfw)
            'model': 'KoboldAI/OPT-2.7B-Nerys-v2',  # model name on huggingface
            "model_gguf_file": "",
            "model_gguf_type": "",
            "model_illustrator": "CompVis/stable-diffusion-v1-4",  # image model name on huggingface
            'illustrator_negative_prompt' : 'poor quality, bad anatomy, empty eyes, ugly, blurry, worst quality, missing limbs, deformed, error, jpeg, text, low res',
            'cputts': True,  # force TTS to run on CPU
            'cputext': True,  # force text generation to run on CPU
            'cpuillustrate': True,  # force image generation to run on CPU
            # possible values are 4, 8, 16 (default 16), lower values reduce VRAM usage
            'language': 'en',
            'save_directory': 'saved_stories',
            'gpu_memory': 1000
        }

    def get(self, key):
        self._create_file_if_not_exists()

        with open(self.file, 'r') as f:
            settings = json.load(f)
            return settings[key] if key in settings else self.defaults[key]

    def set(self, key, value):
        self._create_file_if_not_exists()

        with open(self.file, 'r') as f:
            settings = json.load(f)

        settings[key] = value

        with open(self.file, 'w') as f:
            json.dump(settings, fp=f, indent=4)

    def _create_file_if_not_exists(self):
        if not os.path.isfile(self.file):
            with open(self.file, 'w') as f:
                json.dump(self.defaults, fp=f, indent=4)
