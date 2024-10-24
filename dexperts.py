from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict

class DExpertsLlama:
    def __init__(
        self,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        alpha: float = 0.1,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        chat_template: A dictionary containing the keys 'prefix', 'suffix', and optionally 'system_prompt'
        to define the chat formatting for expert models.
        """

        model_kwargs = model_kwargs or {}
        if expert_model_name_or_path == antiexpert_model_name_or_path:
            self.expert = AutoModelForCausalLM.from_pretrained(
                expert_model_name_or_path, **model_kwargs
            ).to('cuda')
            self.antiexpert = self.expert
        else:
            self.expert = AutoModelForCausalLM.from_pretrained(
                expert_model_name_or_path, **model_kwargs
            ).to('cuda')
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs
            ).to('cuda')
        
        self.expert.half()  
        self.antiexpert.half()

        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.expert.device

    def forward(self, expert_input_ids, antiexpert_input_ids, attention_mask=None, prefix_length=5, return_dict=True):
        # make sure attention_mask exists
        if attention_mask is None:
            attention_mask = torch.ones_like(expert_input_ids)
        antiexpert_attention_mask = attention_mask[:, -prefix_length:]
        
        expert_inputs = {
            "input_ids": expert_input_ids,
            "attention_mask": attention_mask
        }
        antiexpert_inputs = {
            "input_ids": antiexpert_input_ids,
            "attention_mask": antiexpert_attention_mask
        }

        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        del expert_inputs
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return expert_outputs, antiexpert_outputs

    # def _get_tokenized_chat_inputs(self, input_ids, is_expert=True):
    #     """Decode input_ids and encode again to insert chat formatting"""

    #     prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #     chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in prompts] if is_expert else prompts
    #     chat_inputs = self.tokenizer(
    #         chat_prompts, padding="longest", return_tensors="pt",
    #         add_special_tokens=True
    #     )
    #     chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
    #     chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

    #     return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        prefix_length: Optional[int] = 5,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        # make sure input_ids is on the correct device
        input_ids = input_ids.to(self.expert.device)

        # create attention_mask
        attention_mask = torch.ones_like(input_ids)
        
        # initialize expert_input_ids and antiexpert_input_ids
        expert_input_ids = input_ids.clone()
        antiexpert_input_ids = input_ids.clone()
        
        # initialize unfinished_sequences

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # initialize eos_token_id_tensor
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id], device=input_ids.device)

        # initialize expert_kwargs and antiexpert_kwargs
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # initialize analysis_data
        analysis_data = defaultdict(list) if return_logits_for_analysis else None
        expert_input_ids = input_ids.clone()
        #print('expert_input_ids', expert_input_ids.size())
        antiexpert_input_ids = input_ids[:, -prefix_length:]

        for _ in range(max_new_tokens):
            expert_outputs, antiexpert_outputs = self.forward(
                expert_input_ids=expert_input_ids,
                antiexpert_input_ids=antiexpert_input_ids,
                attention_mask=attention_mask,
                prefix_length=prefix_length,
                return_dict=True
            )

            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            # ensure the dimension of expert model's logits is the same as that of the anti-expert model
            expert_next_token_logits = expert_next_token_logits[:, :antiexpert_next_token_logits.shape[-1]]

            # DExperts!
            next_token_logits = (
                expert_next_token_logits - self.alpha * antiexpert_next_token_logits
            )

            # preprocess logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # adjust logits
            if temperature != 1.0 and temperature != 0.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                filtered_logits = torch.topk(next_token_logits, int(top_p * next_token_logits.size(-1)), dim=-1).values.min()
                next_token_logits = torch.where(next_token_logits >= filtered_logits, next_token_logits, torch.tensor(float('-inf')).to(next_token_logits.device))

            # decoding
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                if temperature == 0.0:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # update input_ids
            # print('before add', next_tokens)
            next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            # print('after add', next_tokens)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # print('input_ids', input_ids)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)
            antiexpert_input_ids = input_ids[:, - prefix_length:].to(input_ids.device) if input_ids.size(1) > prefix_length else input_ids

            # update attention_mask
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # 更新 kwargs
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            
            del expert_outputs, antiexpert_outputs

            # update unfinished_sequences
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            #print('unfinished_sequences', unfinished_sequences)

            # check stopping criteria
            if unfinished_sequences.max() == 0 or (stopping_criteria and stopping_criteria(input_ids, None)):
                break

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'expert': expert_next_token_logits,
                    'antiexpert': antiexpert_next_token_logits
                }
                self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs