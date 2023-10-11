from typing import Tuple, List, Optional
import torch 

# --------- Generation Functions ---------
class GenerationItem:
    def __init__(
        self, 
        token_ids: torch.LongTensor, 
        logsum: float, 
        classification_score: Optional[float] = 0, 
        text: Optional[str] = "", 
        num_tokens_generated:Optional[int]=-1, 
        classification_rank: Optional[int]=-1, 
        beamsearch_stopped: Optional[bool]=False,
        seq_score: Optional[float] = 0.0,
        curr_label_idx: Optional[int] = -1, # idx of curr target_label in the target label list
    ):
        self.token_ids = token_ids
        self.logsum = logsum
        self.classification_score = classification_score
        self.text = text
        self.num_tokens_generated = num_tokens_generated
        self.classification_rank = classification_rank
        self.beamsearch_stopped = beamsearch_stopped
        self.seq_score = seq_score
        self.curr_label_idx = curr_label_idx
        # self.prev_logsum = prev_logsum # for beam search span generation

    def get_avg_log(self):
        return self.logsum / self.num_tokens_generated
