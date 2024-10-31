from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch


class Generator:
    def __init__(self,
                 model_name='KoboldAI/OPT-2.7B-Nerys-v2',
                 gpu=True,
                 precision=16,
                 offload_to_memory=False):
        """
        :model_name='KoboldAI/OPT-2.7B-Nerys-v2' : String, which model to use from huggingface
        :gpu: use gpu (default: True)
        :precision: floating point precision
        :offload_to_memory: stores the model in memory when not in use (eat loads of RAM but saves vram)
        """
        self.offload_to_memory = offload_to_memory
        self.device = 'cuda' if gpu else 'cpu'

        if precision == 16:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              torch_dtype=torch.float16 if gpu else torch.float32)
            self.model.to(self.device)
        elif precision == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        elif precision == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        else:
            raise ValueError(f'float precision {precision} not supported')

        self.enc = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)

        self.streamer = TextStreamer(self.enc, skip_prompt=True)

    def generate(self, prompt: str, length, stream=True, eos_tokens=[]):
        eos_token_ids = [self.enc.encode(term)[-1] for term in eos_tokens]

        model_inputs = self.enc([prompt], return_tensors='pt').to(self.device)

        if self.offload_to_memory:
            self.model.to(self.device)

        print('\033[96m', end='')
        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=length,
                do_sample=True,
                use_cache=True,
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
            print(e)
            return ''
        finally:
            print('\033[00m', end='')

            if self.offload_to_memory:
                self.model.to('cpu')

        return self.enc.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:],
                                     clean_up_tokenization_spaces=False)[0]
