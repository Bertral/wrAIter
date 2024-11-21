import json
import os
import re
from difflib import SequenceMatcher
import hashlib
from collections import Counter

import math

from settings import Settings

from generator.generator import Generator
import csv

SAVE_PATH = Settings().get('save_directory')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class Story:
    def __init__(self, gen: Generator, censor: bool):
        self.gen_length = 80
        self.stream = True
        self.censor = censor
        self.gen = gen
        self.events = []
        self.title = ''
        self.save_path = SAVE_PATH

        with open("./story/censored_words.txt", "r") as f:
            self.censored_words = [l.strip(" \n\r,.") for l in f.readlines()]

    def load(self, save_name: str):
        self.title = save_name
        file_name = str(save_name) + ".json"
        exists = os.path.isfile(os.path.join(self.save_path, file_name))
        if exists:
            with open(os.path.join(self.save_path, file_name), "r") as fp:
                self.events = json.load(fp)
            return str(self)
        else:
            return "Error save not found."

    def save(self, save_name: str, name_is_title=True):
        if name_is_title:
            self.title = save_name
        file_name = str(save_name) + ".json"
        with open(os.path.join(self.save_path, file_name), "w") as fp:
            json.dump(self.events, fp, indent=4)


    def new(self, context: str = ''):
        self.title = ''
        self.events = [context + '\n']
        return str(self)

    def get_max_history(self):
        max_tokens = self.gen.model.config.max_position_embeddings
        if hasattr(self.gen.model.config, 'sliding_window'):
            max_tokens = min(self.gen.model.config.sliding_window * 2, max_tokens)
        return max_tokens - self.gen_length

    def clean_input(self, action=''):
        # find the biggest memory that fits max_tokens
        mem_ind = 1
        while len(
                self.gen.enc.encode(self.events[0] + ''.join(filter(None, self.events[-mem_ind:]))
                                    + action)) < self.get_max_history() and len(self.events) - 1 >= mem_ind:
            mem_ind += 1
        mem_ind -= 1

        events_clipped = self.events[0]
        strip_one = True
        while mem_ind > 0:
            if len(self.events) - 1 >= mem_ind and self.events[-mem_ind]:
                if strip_one:
                    events_clipped += self.events[-mem_ind].lstrip(' ')
                    strip_one = False
                else:
                    events_clipped += self.events[-mem_ind]
            mem_ind -= 1

        text = events_clipped + action
        return text

    def clean_result(self, result):
        result = re.sub(rf'{self.gen.enc.eos_token}[\s\S]*$', '', result)  # parse endoftext token that end the text
        result = re.sub(rf'^({self.gen.enc.eos_token})+', '', result)  # remove leading endoftext tokens

        result = result.replace("’", "'")
        result = result.replace("`", "'")
        result = result.replace("“", '"')
        result = result.replace("”", '"')
        result = result.replace(" \n", '\n')
        result = result.replace("\n\n", '\n')

        # remove repeating substrings of 2+ characters at the end of result
        result = re.sub(r'([\s\S]{2,})([\s\S]?\1)+$', r'\1', result)

        # remove sentences that are cut in the middle
        end_of_sentence_index = next(iter([i for i, j in list(enumerate(result, 1))[::-1] if j in '".:?!']),
                                     len(result))
        result = result[:end_of_sentence_index]

        # remove trailing start of quote
        result = re.sub(r'\s"$', '', result)

        if self.censor:
            result = re.sub(r'|'.join(rf'(\b{re.escape(s)}\b)' for s in self.censored_words), '[CENSORED]', result,
                            flags=re.IGNORECASE)

        result = result.rstrip('\n')

        if (str(self) + result).rsplit('\n', maxsplit=1)[-1].count('"') % 2:
            # close quotation marks
            result += '"'

        return result

    def act(self, action: str = '', tries: int = 10, eos_tokens=[]):
        max_tries = tries
        input_str = self.clean_input(action)
        self.events.append(action)
        try:
            result = self.gen.generate(input_str, stream=self.stream, eos_tokens=eos_tokens, length=self.gen_length)
            result = self.clean_result(result)

            while (len(result) < 2
                   or SequenceMatcher(None, self.events[-1], result).ratio() > 0.9 - 0.5 * (tries / max_tries)
                   or '[CENSORED]' in result):
                # reject censored output, empty outputs and repeating outputs (tolerance to repeats increases from 0.4 to
                # 0.9 progressively with each try)
                if tries == 0:
                    return None
                tries -= 1
                result = self.gen.generate(input_str, stream=self.stream, eos_tokens=eos_tokens, length=self.gen_length)
                # print(result)
                result = self.clean_result(result)
        except KeyboardInterrupt as e:
            raise e
        self.events.append(result)
        return result

    def gen_n_results(self, n: int = 3, tries: int = 30):
        res = []
        max_tries = tries
        input_str = self.clean_input()

        while len(res) < n and tries > 0:
            result = self.gen.generate(input_str, stream=False, length=self.gen_length,
                                       eos_tokens=['.', '!', '?', '\n'])
            result = self.clean_result(result)

            if not (len(result) < 2
                    or SequenceMatcher(None, self.events[-1], result).ratio() > 0.9 - 0.5 * (tries / max_tries)
                    or '[CENSORED]' in result
                    or result in res):
                res.append(result)

            tries -= 1

        return res

    def wordcloud(self, top_n=10, include_count=False, history_lookback=None):
        def str_to_words(text):
            return [w for w
                    in re.sub(r"[\W\d]+", ' ', text.lower().replace("n't", '')).split()
                    if len(w) > 1]

        words = str_to_words(str(self))

        if history_lookback:
            words = words[-history_lookback:] + str_to_words(self.events[0])
        wordcloud = {w: count for w, count in Counter(words).most_common() if count > 0}

        # adjust weight relatively to usual frequency of words
        min_freq = 1e20
        found_words = set()
        with open('story/unigram_freq_by_Rachael_Tatman_on_kaggle.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                if row['word'] in wordcloud:
                    wordcloud[row['word']] = math.log(wordcloud[row['word']]/int(row['count']))
                    min_freq = min(int(row['count']), min_freq)
                    found_words.add(row['word'])

        # assume unknown words have low frequency, but still have a frequency
        for w in wordcloud.keys() - found_words:
            wordcloud[w] = math.log(wordcloud[w] / min_freq)

        wordcloud = sorted(wordcloud.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if include_count:
            return ', '.join([f'{w} ({n})' for w, n in wordcloud])
        else:
            return ', '.join([w for w, _ in wordcloud])

    def __str__(self):
        text = ''.join(filter(None, self.events)).lstrip()
        return text
