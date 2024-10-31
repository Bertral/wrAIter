import json
import os
import re

from generator.generator import Generator
from story.story import Story

class Conversation(Story):
    def __init__(self, gen: Generator, censor: bool):
        super().__init__(gen, censor)
        self.gen_length = 500
        self.player = 'Me'
        self.bot = 'Bot'

    def load(self, save_name: str):
        self.title = save_name[:-15]
        file_name = str(save_name) + ".json"
        exists = os.path.isfile(os.path.join(self.save_path, file_name))
        if exists:
            with open(os.path.join(self.save_path, file_name), "r") as fp:
                j = json.load(fp)
                self.events = j['events']
                self.bot = j['bot']
                self.player = j['player']
            return str(self)
        else:
            return "Error save not found."

    def save(self, save_name: str, name_is_title=True):
        if name_is_title:
            self.title = save_name
        file_name = str(save_name) + " (conversation).json"
        with open(os.path.join(self.save_path, file_name), "w") as fp:
            json.dump({'type': 'conversation', 'player':self.player, 'bot': self.bot, 'events': self.events}, fp, indent=4)

    def act(self, action: str = '', tries: int = 10, eos_tokens=[]):
        return super().act(action, tries, ['"', '?"', '!"', '."', '\n'] + eos_tokens)

    def new(self, context: str = '', player='Me', bot='Bot'):
        self.player = player
        self.bot = bot
        return super().new(context)

    def clean_result(self, result):
        result = re.sub(rf'(\n|"|{self.gen.enc.eos_token})[\s\S]*$', '', result)  # parse endoftext token that end the text
        result = super().clean_result(result)
        if result and result[-1] not in ['.', '!', '?', '"']:
            result += '.'
        if result and result[-1] not in ['"'] and f'[{self.bot}]' not in self.events[-1]:  # don't add " for french grammar
            result += '"'
        return result

