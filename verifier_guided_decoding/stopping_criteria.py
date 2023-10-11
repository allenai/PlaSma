from transformers.generation_stopping_criteria import StoppingCriteria
from nltk import sent_tokenize 
import torch
import re
from utils import is_sent_complete


class EndSentenceCriteria(StoppingCriteria):
    """
    Only suitable for single instance
    stop generation after generation of each sentence

    Args:
        num_sents: stop generation after specified number of sentences is generated   # deprecate
        
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # self.generated_sents = 0

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.size(0) == 1 # EndSentenceCriteria only works for batch size 1
        text = self.tokenizer.decode(input_ids[0])
        if is_sent_complete(text):
            return True
        return False


class MultiBatchEndSentenceCriteria(StoppingCriteria):
    """
    stop generation after generation if all last tokens is pad_token_ids

    Args:
    
        num_sents: stop generation after specified number of sentences is generated   # deprecate
        
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor) -> bool:
        # import pdb; pdb.set_trace()
        return (input_ids[:,-1]==self.pad_token_id).all().item()

class EndSpanCriteria(StoppingCriteria): # stop the generation after n x span_size generations
    """
    stop generation after generation after a specified span length is generated

    Args:
        stop_by_tokens: stop generation after specified number of tokens is generated 
    """
    def __init__(self, stop_by_tokens):
        self.stop_by_tokens = stop_by_tokens

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor) -> bool:
        return (input_ids.size(-1) >=self.stop_by_tokens)

class MultiBatchEndStepCriteria(StoppingCriteria):
    """
    stop generation after generation if all last tokens is pad_token_ids

    Args:
    
        num_sents: stop generation after seeing "Step"
        
    """
    # Faeze TODO: how to stop furthur generations for examples who stoped?
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token = self.tokenizer.encode("Step")[-2] # for BART need to take the 1st elemnt, as 0 and 2nd are boes and eos tokens

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor) -> bool:
        # import pdb; pdb.set_trace()
        return (input_ids[:,-1]==self.stop_token).all().item()