import torch
from accelerate import cpu_offload
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers.cache_utils import DynamicCache


class Generator:
    def __init__(self,
                 model_name='KoboldAI/OPT-2.7B-Nerys-v2',
                 model_gguf_file=None,
                 model_gguf_type=None,
                 gpu=True,
                 gpu_memory=1000):
        """
        :model_name='KoboldAI/OPT-2.7B-Nerys-v2' : String, which model to use from huggingface
        :gpu: use gpu (default: True)
        :precision: floating point precision
        :offload_to_memory: stores the model in memory when not in use (eat loads of RAM but saves vram)
        """
        self.device = 'cuda' if gpu else 'cpu'
        self.model_name = model_name

        if self.device == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto',
                                                              max_memory={0: f'{gpu_memory}GiB', 'cpu': '1000GiB'},
                                                              attn_implementation='eager',
                                                              gguf_file=model_gguf_file if model_gguf_file else None,
                                                              model_type=model_gguf_type if model_gguf_type else None)

        self.enc = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False,
                                                 gguf_file=model_gguf_file if model_gguf_file else None,
                                                 model_type=model_gguf_type if model_gguf_type else None)
        self.streamer = TextStreamer(self.enc, skip_prompt=True)

    def generate(self, prompt: str, length, stream=True, eos_tokens=[]):
        print('\033[96m', end='')
        eos_token_ids = [self.enc.encode(term)[-1] for term in eos_tokens]

        try:
            model_inputs = self.enc([prompt], return_tensors='pt').to(self.device)
            cache = DynamicCache() if 'Mistral' in self.model_name else None
            # generate in segments to build up a cache and never spike memory usage
            caching_tokens = 500
            for i in range(caching_tokens, len(model_inputs['input_ids'][0]), caching_tokens):
                cache = self.model.generate(
                    input_ids=model_inputs['input_ids'][:, :i],
                    attention_mask=model_inputs['attention_mask'][:, :i],
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=cache,
                    pad_token_id=self.enc.eos_token_id,
                    eos_token_id=eos_token_ids + [self.enc.eos_token_id]
                ).past_key_values

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=length,
                do_sample=True,
                use_cache=True,
                past_key_values=cache,
                pad_token_id=self.enc.eos_token_id,
                streamer=self.streamer if stream else None,
                repetition_penalty=1.05,
                eos_token_id=eos_token_ids + [self.enc.eos_token_id]
            )
        except KeyboardInterrupt as e:
            self.streamer.end()
            raise e
        except RuntimeError as e:
            self.streamer.end()
            if not isinstance(e, torch.cuda.OutOfMemoryError):
                raise e

            print(f'Out of GPU memory: offloading layers to CPU')
            cpu_offload(self.model)
            torch.cuda.empty_cache()
            return ''
        finally:
            print('\033[00m', end='')
        return self.enc.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:],
                                     clean_up_tokenization_spaces=False)[0]

    def extract_gender(self, prompt):
        """
        Returns whether "he", "she" or "I/you/they" is the most likely token that follows the prompt.
        :param prompt:
        :return: str, one of ('m', 'f', 'nb')
        """

        model_inputs = self.enc([prompt], return_tensors='pt').to(self.device)

        # we assume that space+pronoun is encoded into a single token
        tokens = {
            self.enc.encode(' He')[-1]: 'm', self.enc.encode(' he')[-1]: 'm',
            self.enc.encode(' She')[-1]: 'f', self.enc.encode(' she')[-1]: 'f',
            self.enc.encode(' They')[-1]: 'nb', self.enc.encode(' they')[-1]: 'nb',
            self.enc.encode(' You')[-1]: 'nb', self.enc.encode(' you')[-1]: 'nb',
            self.enc.encode(' I')[-1]: 'nb'
        }

        # cap input at 512 tokens, the full history would be overkill
        scores = self.model.generate(
            input_ids=model_inputs['input_ids'][:, -512:],
            attention_mask=model_inputs['attention_mask'][:, -512:],
            max_new_tokens=1,
            pad_token_id=self.enc.eos_token_id,
            eos_token_id=[self.enc.eos_token_id],
            output_scores=True, return_dict_in_generate=True
        ).scores[-1]
        output_probs = scores.softmax(1).mean(axis=0).flatten().cpu().detach().numpy()
        token_prob = {k : output_probs[k] for k in tokens.keys()}
        return tokens[max(token_prob, key=token_prob.get)]

