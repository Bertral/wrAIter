import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from settings import Settings
from story.story import Story


class Illustrator:
    HEIGHT = 512
    WIDTH = 512

    def __init__(self, settings: Settings):
        self.device = 'cuda' if not settings.get('cpuillustrate') else 'cpu'
        self.model_name = settings.get('model_illustrator')
        self.negative_prompt = settings.get('illustrator_negative_prompt')
        self.censor = settings.get('censor')

        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name,
                                                             torch_dtype=torch.float32)
        if not self.censor:
            self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    def illustrate(self, story: Story):
        input = story.wordcloud(top_n=15, history_lookback=500)
        print('keywords:', input)

        self.pipe = self.pipe.to(self.device)
        image = self.pipe(
            input,
            height=self.HEIGHT,
            width=self.WIDTH,
            negative_prompt=self.negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.0,
        ).images[0]
        self.pipe = self.pipe.to('cpu')
        image.show(title='wraiter display')
