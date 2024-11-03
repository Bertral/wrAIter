#!/usr/bin/env python3
import os

from InquirerPy.separator import Separator
from prompt_toolkit.filters import Condition

import settings
from illustrator.illustrator import Illustrator
from postprocess import postprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from audio.tts import Dub
from InquirerPy import inquirer
from story.story import Story
from story.story import SAVE_PATH as SAVE_PATH
from story.conversation import Conversation
from generator.generator import Generator
import re
import readline  # actually necessary for pyinquirer to work consistently
import shutil
from textwrap import TextWrapper


class Game:
    def __init__(self):
        self.settings = settings.Settings()
        self.status_text = 'Press Enter twice to send, Ctrl-L for a list of commands'
        self.gen = Generator(model_name=self.settings.get('model'), gpu=not self.settings.get('cputext'),
                             gpu_memory=self.settings.get('gpu_memory'))
        self.tts = None if self.settings.get('silent') else Dub(gpu=not self.settings.get('cputts'),
                                                                lang=self.settings.get('language'))
        self.stt = None
        self.illustrator = None
        self.story = Story(self.gen, censor=self.settings.get('censor'))
        self.loop = self.loop_text
        self.sample_hashes = []
        self.keybind_pressed = None
        self.redo_history = []

    def play(self):
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')

            width = shutil.get_terminal_size(fallback=(82, 40)).columns
            print("▄" * width)
            print("░███░█░▄▄▀█░▄▄▀█▄░▄█▄░▄█░▄▄█░▄▄▀".center(width, '█'))
            print("▄▀░▀▄█░▀▀▄█░▀▀░██░███░██░▄▄█░▀▀▄".center(width, '█'))
            print("█▄█▄██▄█▄▄█░██░█▀░▀██▄██▄▄▄█▄█▄▄".center(width, '█'))
            print("▀" * width)

            choices = []

            most_recent_story = None
            for path, subdirs, files in os.walk(SAVE_PATH):
                for name in reversed(
                        sorted(files, key=lambda name: os.path.getmtime(os.path.join(path, name)))):
                    if name.startswith('__autosave__'):
                        continue
                    if name.endswith('.json'):
                        most_recent_story = name[:-5]
                        break

            continue_choice = 'continue'
            if self.story.events:
                choices.append(continue_choice)
            elif most_recent_story is not None:
                continue_choice = f'continue "{most_recent_story}"'
                choices.append(continue_choice)

            if [f for f in os.listdir(SAVE_PATH) if f.endswith('.json')]:
                choices.append('load')

            choices.append('new story')
            choices.append('new conversation')
            choices.append(Separator(f'{" settings " :=^22}'))

            choices.append('unmute audio' if self.tts is None else 'mute audio')

            choices.append('allow profanity' if self.settings.get('censor') else 'censor profanity')

            if self.loop != self.loop_text:
                choices += ['switch to text input']
            if self.loop != self.loop_choice and not isinstance(self.story, Conversation):
                choices += ['switch to choice input']
            if self.loop != self.loop_voice:
                choices += ['switch to voice input']

            if len(self.story.events) > 1:
                choices.insert(1, 'save')

            action = inquirer.select(f'{" data " :=^22}',
                                     choices=choices,
                                     raise_keyboard_interrupt=False).execute()

            if action == 'new story':
                self.new_prompt()
            if action == 'new conversation':
                self.new_prompt(conv=True)
            elif action == continue_choice:
                if len(self.story.events) == 0:
                    if most_recent_story.endswith(' (conversation)'):
                        self.story = Conversation(self.gen, censor=self.settings.get('censor'))
                    else:
                        self.story = Story(self.gen, censor=self.settings.get('censor'))
                    self.story.load(most_recent_story)
            elif action == 'save':
                self.save_prompt()
            elif action == 'load':
                self.load_prompt()
            elif action == 'switch to choice input':
                self.stt = None
                self.loop = self.loop_choice
            elif action == 'switch to text input':
                self.stt = None
                self.loop = self.loop_text
            elif action == 'switch to voice input':
                from audio.stt import CustomMic
                self.stt = CustomMic(english=(self.settings.get('language') == 'en'), model='medium')
                self.loop = self.loop_voice
            elif action in ('mute audio', 'unmute audio'):
                if self.tts is None:
                    self.tts = Dub(gpu=not self.settings.get('cputts'), lang=self.settings.get('language'))
                    self.settings.set('silent', False)
                else:
                    self.tts.stop()
                    self.tts = None
                    self.settings.set('silent', True)
            elif action == 'censor profanity':
                self.story.censor = True
                self.settings.set('censor', True)
            elif action == 'allow profanity':
                self.story.censor = False
                self.settings.set('censor', False)
            else:
                print('invalid input')

            if len(self.story.events) > 0:
                self.loop()

    def new_prompt(self, conv=False):
        if not conv:
            context = inquirer.text("Type a short context that the AI won't forget, so "
                                    "preferably describe aspects\n"
                                    "of the setting that you expect to remain true as the story develops."
                                    "\nWho are your characters? What world do they live in?"
                                    "\nYou can also use a tagging syntax like [Genre: fantasy, horror] for example"
                                    "\nor leave it empty.",
                                    qmark='', amark='', raise_keyboard_interrupt=False,
                                    multiline=True, instruction=' ', mandatory=False,
                                    long_instruction='Press Enter twice to send, Ctrl-C to cancel',
                                    keybindings={'answer': [{'key': ['enter', 'enter']}]}).execute()
            if context is None:
                return

            self.story = Story(self.gen, censor=self.settings.get('censor'))
            self.story.new(context)
        else:
            player = inquirer.text("Enter your tag. It can be 'You', 'A', 'B', 'C', 'Me', your name, ...\n>",
                                   qmark='', amark='', raise_keyboard_interrupt=False, default='Me', mandatory=False,
                                   long_instruction='Ctrl-C to cancel').execute()
            if player is None:
                return
            bot = inquirer.text("Enter the AI's tag. It can be 'Them', 'Him', 'Her', 'Bot', their name, ...\n>",
                                qmark='', amark='', raise_keyboard_interrupt=False, default='Bot', mandatory=False,
                                long_instruction='Ctrl-C to cancel').execute()
            if bot is None:
                return
            context = inquirer.text("Type a short context about who you're talking to "
                                    "(or don't and it will be random).\n"
                                    'For example: "The following conversation happens after a car crash."',
                                    qmark='', amark='', raise_keyboard_interrupt=False,
                                    multiline=True, instruction=' ', mandatory=False,
                                    long_instruction='Press Enter twice to send, Ctrl-C to cancel',
                                    keybindings={'answer': [{'key': ['enter', 'enter']}]}).execute()
            if context is None:
                return

            self.story = Conversation(self.gen, censor=self.settings.get('censor'))
            self.story.new(context, player=player.strip(), bot=bot.strip())

    def load_prompt(self):
        action = inquirer.select('Choose a file to load',
                                 choices=sorted(
                                     [f[:-5] for f in os.listdir(SAVE_PATH) if f.endswith('.json')],
                                     key=lambda name: os.path.getmtime(os.path.join(SAVE_PATH, name + '.json')),
                                     reverse=True),
                                 raise_keyboard_interrupt=False, mandatory=False,
                                 long_instruction='Ctrl-C to cancel').execute()

        if action is not None:
            if action.endswith(' (conversation)'):
                self.story = Conversation(self.gen, censor=self.settings.get('censor'))
            else:
                self.story = Story(self.gen, censor=self.settings.get('censor'))
            self.story.load(action)

    def save_prompt(self):
        user_input = inquirer.text("Type a name for your save file.\n>",
                                   default=self.story.title, mandatory=False,
                                   qmark='', amark='', raise_keyboard_interrupt=False).execute()
        if user_input is not None:
            user_input = user_input.strip()
            try:
                self.story.save(user_input)
                self.status_text = f'Successfully saved {user_input}'
            except:
                self.status_text = f'Failed to save as {user_input}'

    def loop_text(self):
        self.pprint()

        while True:
            try:
                self.pprint()
                default_input = self.redo_history[0] if self.redo_history else ''
                if self.redo_history:
                    self.status_text = f'[{len(self.story.events)}/{len(self.redo_history) + len(self.story.events) - 1}]'
                inquirer_prompt = inquirer.text(message='', qmark='', amark='', raise_keyboard_interrupt=False,
                                                mandatory=False, default=default_input, multiline=True,
                                                long_instruction=self.status_text,
                                                instruction=' ', keybindings={'answer': [{'key': ['enter', 'enter']}]})
                self.status_text = 'Enter twice to send, Ctrl-L for commands'

                # Declare keybinds
                @inquirer_prompt.register_kb('escape')
                def menu(event):
                    self.keybind_pressed = menu
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('c-w')
                def wordstats(event):
                    tokens_current = len(self.story.gen.enc.encode(str(self.story)))
                    tokens_max = self.story.get_max_history()
                    tokens_trimmed = len(self.story.gen.enc.encode(self.story.clean_input()))
                    story_trimmed = self.story.clean_input()[len(self.story.events[0]):][:160].lstrip('\n')
                    fancy_words = ", ".join(sorted(set(re.sub(
                        r"[^A-Za-z0-9_]+", " ",
                        str(self.story).lower()).split()), key=lambda x: len(x), reverse=True)[:10])

                    width = shutil.get_terminal_size(fallback=(82, 40)).columns
                    width = min(width, 180)
                    wrapper = TextWrapper(width=width, replace_whitespace=False, initial_indent='',
                                          subsequent_indent=' ' * 18)

                    body = (f'\n\n'
                            f'story title:      "{self.story.title}"\n'
                            f'wordcloud:        {self.story.wordcloud(top_n=20)}\n'
                            f'fanciest words:   {fancy_words}\n'
                            f'number of events: {len(self.story.events)}\n'
                            f'number of tokens: {tokens_current}/{tokens_max} (trimmed to {tokens_trimmed})\n'
                            f'context/starting prompt:\n'
                            f'                  {self.story.events[0]}\n'
                            f'oldest non-context that fits the input:\n'
                            f'                  {story_trimmed} [...]\n')
                    print('\n'.join(['\n'.join(wrapper.wrap(line)) for line in body.splitlines()]), flush=True)

                    input('Press enter to continue.')
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('c-p')
                def edit_story_prompt(event):
                    self.keybind_pressed = edit_story_prompt
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('tab')
                def tab(event):
                    self.keybind_pressed = tab
                    inquirer_prompt._handle_enter(event)

                @inquirer_prompt.register_kb('c-s')
                def save(event):
                    self.keybind_pressed = save
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('c-o')
                def illustrate(event):
                    self.keybind_pressed = illustrate
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('up', filter=Condition(lambda: len(self.story.events) > 1))
                def revert(event):
                    if self.tts is not None:
                        self.tts.stop()
                    undo = 2 if isinstance(self.story, Conversation) else 1
                    self.redo_history = self.story.events[-undo:] + self.redo_history
                    self.story.events = self.story.events[:-undo]
                    inquirer_prompt._handle_skip(event)

                @inquirer_prompt.register_kb('down', filter=Condition(lambda: bool(self.redo_history)))
                def redo(event):
                    self.keybind_pressed = redo
                    inquirer_prompt._handle_enter(event)

                @inquirer_prompt.register_kb('c-l')
                def command_list(event):
                    print('\n\nKnown commands:\n'
                          'esc          go to main menu\n'
                          'tab          generate a single sentence\n'
                          'UP   arrow   undo and edit last action or response\n'
                          'DOWN arrow   redo the last undone action\n'
                          'ctrl-l       display this list\n'
                          'ctrl-p       edit context/starting prompt\n'
                          'ctrl-s       save story\n'
                          'ctrl-w       print word count and other stats\n'
                          'ctrl-o       show illustration for current story key words\n'
                          'ctrl-c       reset current text box, interrupt generation and audio\n')
                    input('Press enter to continue.')
                    inquirer_prompt._handle_skip(event)

                # execute inquirer prompt
                self.keybind_pressed = None
                user_input = inquirer_prompt.execute()

                # Handle keybinds that require additional processing
                if self.keybind_pressed == menu:
                    return
                elif self.keybind_pressed == edit_story_prompt:
                    new_context = inquirer.text('Edit context/starting prompt:',
                                                default=self.story.events[0],
                                                qmark='', amark='', raise_keyboard_interrupt=False,
                                                mandatory=False, multiline=True, instruction=' ',
                                                long_instruction='Press Enter twice to send, Ctrl-C to cancel',
                                                keybindings={'answer': [{'key': ['enter', 'enter']}]}).execute()
                    if new_context is not None and new_context != self.story.events[0]:
                        self.story.events[0] = new_context

                elif self.keybind_pressed == save:
                    self.save_prompt()
                elif self.keybind_pressed == illustrate:
                    if self.illustrator is None:
                        self.illustrator = Illustrator(self.settings)
                    self.illustrator.illustrate(self.story)
                elif user_input is None:
                    # CTRL+C case (inquirer returned None)
                    if self.tts is not None:
                        self.tts.stop()
                elif self.keybind_pressed == redo:
                    n = 2 if isinstance(self.story, Conversation) else 1
                    self.story.events.append(user_input)
                    if n > 1:
                        self.story.events += self.redo_history[1:n]
                    self.redo_history = self.redo_history[n:]
                    user_input = None  # skip generation

                # handle text input
                if user_input is not None:
                    self.story.save('__autosave__', name_is_title=False)
                    self.redo_history = []

                    action = user_input.strip(' ')

                    if action and str(self.story)[-1:] != '\n':
                        action = ' ' + action

                    # capitalize
                    action = re.sub(r'\bi\b', 'I', action)  # capitalize lone 'I'
                    action = re.sub(r'[.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)',
                                    lambda matchobj: matchobj.group(0).upper(), action)  # capitalize start of sentences
                    # remove spaces after carriage returns
                    action = re.sub(r' *\n *', '\n', action)

                    if isinstance(self.story, Conversation):
                        if self.settings.get('language') == 'fr':
                            action = f'\n[{self.story.player}:] {action.strip()}\n[{self.story.bot}:] '
                        else:
                            action = f'\n{self.story.player}: "{action.strip()}"\n{self.story.bot}: "'

                    self.pprint(action)

                    eos_tokens = ['.', '!', '?', '\n'] if self.keybind_pressed == tab else []
                    result = self.story.act(action, eos_tokens=eos_tokens)
                    result = postprocess.post_txt2txt(result, self.story)
                    self.pprint()
                    if result is None:
                        print(
                            "--- The model failed to produce an decent output after multiple tries. Try something else.")
                    else:
                        if self.tts is not None:
                            if isinstance(self.story, Conversation):
                                self.tts.deep_play(result)
                            else:
                                self.tts.deep_play(action + result)

                        postprocess.post_tts(self.story)
            except KeyboardInterrupt:
                if self.tts is not None:
                    self.tts.stop()

    def loop_voice(self):
        self.pprint()

        while True:
            try:
                self.story.save('__autosave__', name_is_title=False)
                self.pprint()
                try:
                    user_input = self.stt.custom_listen()
                except KeyboardInterrupt:
                    return

                action = user_input.strip()
                if action and str(self.story)[-1:] != '\n':
                    action = ' ' + action

                # capitalize
                action = re.sub(r'\bi\b', 'I', action)  # capitalize lone 'I'
                action = re.sub(r'^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)',
                                lambda matchobj: matchobj.group(0).upper(), action)  # capitalize start of sentences

                if isinstance(self.story, Conversation):
                    if self.settings.get('language') == 'fr':
                        action = f'\n[{self.story.player}:] {action.strip()}\n[{self.story.bot}:] '
                    else:
                        action = f'\n{self.story.player}: "{action.strip()}"\n{self.story.bot}: "'

                self.pprint(action)

                result = self.story.act(action)
                self.pprint()
                if result is None:
                    print("--- The model failed to produce an decent output after multiple tries. Try something else.")
                else:
                    self.tts.deep_play(result)
            except KeyboardInterrupt as e:
                if self.tts is not None:
                    self.tts.stop()

    def loop_choice(self):
        self.pprint()

        while True:
            try:
                self.story.save('__autosave__', name_is_title=False)
                if isinstance(self.story, Conversation):
                    self.loop = self.loop_text
                    return

                self.pprint()
                results = self.story.gen_n_results(3)
                results = {r.strip('\n').split('\n')[0]: r for r in results}
                choices = list(results.keys()) + ['< revert >', '< more >']

                user_input = inquirer.select('\nChoice:', amark='', qmark='',
                                             choices=choices,
                                             raise_keyboard_interrupt=False, mandatory=False,
                                             long_instruction='Ctrl-C for menu').execute()
                if user_input is None:
                    return
                elif user_input == '< more >':
                    continue
                elif user_input == '< revert >':
                    if self.tts is not None:
                        self.tts.stop()
                    if len(self.story.events) < 4:
                        self.story.new(self.story.events[0])
                        self.pprint()
                        if self.tts is not None:
                            self.tts.deep_play('\n'.join(filter(None, self.story.events[2:])))
                    else:
                        self.story.events = self.story.events[:-1]
                        self.pprint()
                else:
                    user_input = results[user_input]
                    self.story.events.append(user_input)
                    self.pprint()
                    if self.tts is not None:
                        self.tts.deep_play(user_input)
            except KeyboardInterrupt:
                if self.tts is not None:
                    self.tts.stop()

    def pprint(self, highlighted=None):
        width = shutil.get_terminal_size(fallback=(82, 40)).columns
        width = min(width, 180)
        wrapper = TextWrapper(width=width, replace_whitespace=False, initial_indent='  ', subsequent_indent='  ')

        os.system('cls' if os.name == 'nt' else 'clear')

        if highlighted is None:
            highlighted = self.story.events[-1]
            body = ''.join(filter(None, self.story.events[:-1])).lstrip()
        else:
            body = str(self.story)

        body = f'{body}\033[96m{highlighted}\033[00m'

        print('\n'.join(['\n'.join(wrapper.wrap(line)) for line in body.splitlines()]),
              end='', flush=True)


if __name__ == "__main__":
    g = Game()
    g.play()
