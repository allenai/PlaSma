import os
# below commented off, add in execution script
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # need to appear before importing torch modules; if changed, need to restart kernel
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from typing import Tuple, List, Optional, Union
import argparse
from tqdm import tqdm
from termcolor import colored
import math 
import re
import numpy as np

import pandas as pd
import torch

from stopping_criteria import ( # self-defined
    EndSentenceCriteria,
    EndSpanCriteria,
    MultiBatchEndSentenceCriteria,
)


from generation_utils import GenerationMixin # import generate
from proto import GenerationItem
from utils import clean_input_for_critic, is_valid_step


# --------- Classification Functions ---------
# def get_classification_logprob(model, sentence1, sentence2_1, sentence2_2, label_idx, device=None):
#     """
#     Args:
#         model: classification model
#         sentence1 (str): goal of the form `<source_prefix>: <goal>`
#         sentence2_1: scripts generated so far of the form Step 1: <> Step 2: <> ... Step i: <>
#         sentence2_2: newly generated step usually of the form Step i+1: <>
#         label_idx: desired label
#     returns a tuple of:
#         logprob: log probability of how well the steps logically follow the orevious steps given the goal
#         rank: out of all possible classes, what rank is the given class (the lower the rank, the more likely)
#     """
#     # NOTE: roberta only accepts up to 512 tokens

#     process_1 = lambda x: ". ".join(re.split("Step \d+:", x)[1:]).strip().lower()
#     process_2 = lambda x: re.sub("Step|\d+:", "",x).strip().lower()
    
#     # sentence 1 contains source_prefix which is "provide steps:" for script generation task (TODO: add prefix_source)
#     sentence1 = sentence1.replace("provide steps:", "").lower().strip()

#     # s
#     sentence2_1, sentence2_2 = process_1(sentence2_1) if sentence2_1 != "Step 1:" else "", process_2(sentence2_2)
#     all_probs = model(f"goal: {sentence1}. {sentence2_1}</s></s>{sentence2_2}")[0]
#     label_prob = [(ex['label'], ex['score']) for ex in all_probs]
#     label_prob = sorted(label_prob, key = lambda x: x[1], reverse=True)
#     rank, logprob = [(i, math.log(val[1])) for i, val in enumerate(label_prob) if val[0] == label_idx][0]
#     # logprob = math.log([ex['score'] for ex in all_probs if ex['label'] == label_idx])

#     return (logprob, rank)

# --------- Classification Functions ---------
def get_classification_logprob(model, sentence1, sentence2_1, sentence2_2, label_idx="good", device=None):
    """
    Args:
        model: classification model
        sentence1 (str): goal of the form `<source_prefix>: <goal>`
        sentence2_1: scripts generated so far of the form Step 1: <> Step 2: <> ... Step i: <>
        sentence2_2: newly generated step usually of the form Step i+1: <>
        label_idx: desired label
    returns a tuple of:
        logprob: log probability of how well the steps logically follow the orevious steps given the goal
        rank: out of all possible classes, what rank is the given class (the lower the rank, the more likely)
    """

    # process_1 = lambda x: ". ".join(re.split("Step \d+:", x)[1:]).strip().lower() # [1:]
    process_1 = lambda x: ". ".join(re.split("\s*Step \d+: ", x)[1:]).strip().lower()[:-len(" step")] # [1:]
    process_2 = lambda x: re.sub("Step|\d+:", "",x).strip().lower()
    # import pdb; pdb.set_trace()

    sentence1 = clean_input_for_critic(sentence1)
    sentence2_1, sentence2_2 = process_1(sentence2_1) if sentence2_1 != "Step 1:" else "", process_2(sentence2_2)
    all_probs = model(f"goal: {sentence1}. {sentence2_1}</s></s>{sentence2_2}")[0]
    label_prob = [(ex['label'], ex['score']) for ex in all_probs]
    label_prob = sorted(label_prob, key = lambda x: x[1], reverse=True)
    rank, logprob = [(i, math.log(val[1])) for i, val in enumerate(label_prob) if val[0] == label_idx][0]


    return (logprob, rank)
    

# # --------- Generation Functions ---------
def process_beamsearch_generation(tokenizer, input, beamsearch_outputs, start_pos, classification_model, prev_gen: Optional[GenerationItem] = None, target_label="good", decoder_input_ids=None):
    """
    return:
        tuple (GenerationItem, beamsearch_stopped)
        if no additional sentence is generated, returned GenerationItem is None, beamsearch_stopped is True 
    """
    # NOTE: 2 dim, with first dim of size 1 in order to work
    assert beamsearch_outputs.sequences.size(0) == 1 and beamsearch_outputs.sequences.dim() == 2
    # cut off pad ids
    pad_mask = beamsearch_outputs.sequences==tokenizer.pad_token_id
    eos_mask = beamsearch_outputs.sequences==tokenizer.eos_token_id
    comb_mask = pad_mask.logical_or(eos_mask)
    # assert comb_mask.size(0) == 1 and comb_mask.dim() == 2
    last_valid_idx = (comb_mask == False).nonzero()[-1][1].item() 
    curr_gen_ids = beamsearch_outputs.sequences[:,:last_valid_idx+1]
    # only add to generations if new sentence generated
    # future beam search will be skipped if no new sentence generated from previous beam search
    # if prev_gen is None or curr_gen_ids.size(1) > prev_gen.token_ids.size(1):
    # import pdb; pdb.set_trace()
    if prev_gen is None or curr_gen_ids.size(1) > start_pos: # + 1: # + 1 added 
        # get scores
        # start_pos = 1 if prev_gen is None else prev_gen.token_ids.size(1)
        end_pos = last_valid_idx+1 # later put pad token probability to be 1 
        gen_ids = beamsearch_outputs.sequences[:, start_pos:end_pos][0]
        num_tokens_generated = end_pos - start_pos
        logsum = beamsearch_outputs.sequences_scores.item() * (num_tokens_generated**2) # tensor([float])
        new_sent = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # finalize value with prev_gen
        prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 3
        prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
        prev_gen_text = prev_gen.text if prev_gen is not None else tokenizer.decode(decoder_input_ids[0][1:]) if decoder_input_ids is not None else ""
        text = " ".join([prev_gen_text.strip(), new_sent.strip()]) 
        logsum += prev_gen_logsum
        num_tokens_generated += prev_gen_num_tokens

        # classification score
        classification_score, rank = get_classification_logprob(classification_model, input, prev_gen_text, new_sent, target_label) # prev_gen.text contains all previous generations

        item = GenerationItem(curr_gen_ids, logsum, classification_score, text, num_tokens_generated, classification_rank=rank, beamsearch_stopped=False)
        return (item, False) # 
    else:
        return (None, True)

def process_multisample_generation(tokenizer, input, sample_outputs, start_pos, classification_model, prev_gen: Optional[GenerationItem] = None, target_label="good", decoder_input_ids=None):
    """
    return:
        List [GenerationItem]
    """
    # NOTE: sample_outputs size [num_return_sequences, seq_len]
    generations = []

    # cut off pad ids
    pad_mask = sample_outputs.sequences==tokenizer.pad_token_id
    eos_mask = sample_outputs.sequences==tokenizer.eos_token_id
    comb_mask = pad_mask.logical_or(eos_mask)
    probs = torch.stack(sample_outputs.scores, dim=0).softmax(-1) # size [seq_len, num_seq, vocab]

    ### TODO: remove duplicate generation samples (speed up)

    # format each sequence into a GenerationItem
    for num_seq in range(sample_outputs.sequences.size(0)): 
        last_valid_idx = (comb_mask[num_seq] == False).nonzero()[-1].item() 
        curr_gen_ids = sample_outputs.sequences[num_seq,:last_valid_idx+1].unsqueeze(0)
        # get scores
        # start_pos = 1 if prev_gen is None else prev_gen.token_ids.size(1)
        end_pos = last_valid_idx+1 # later put pad token probability to be 1 
        gen_ids = sample_outputs.sequences[num_seq, start_pos:end_pos]
        num_tokens_generated = end_pos - start_pos
        curr_probs = probs[:, num_seq, :].squeeze(1)[:num_tokens_generated] # size [seq_len, num_beams, vocab_size]
        gen_probs = torch.gather(curr_probs, -1, gen_ids[:, None]).squeeze(-1)
        logsum = torch.sum(torch.log(gen_probs)).item() # scalar
        new_sent = tokenizer.decode(gen_ids, skip_special_tokens=True)


        # finalize value with prev_gen
        prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 3
        prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
        prev_gen_text = prev_gen.text if prev_gen is not None else tokenizer.decode(decoder_input_ids[0][1:]) if decoder_input_ids is not None else ""
        text = " ".join([prev_gen_text.strip(), new_sent.strip()]).strip() 
        # logsum += prev_gen_logsum
        # num_tokens_generated += prev_gen_num_tokens

        # classification_score, rank = get_classification_logprob(classification_model, input, prev_gen_text , new_sent, target_label)
        # item = GenerationItem(curr_gen_ids, logsum, classification_score, text, num_tokens_generated, classification_rank = rank)
        
        # Faeze
        if is_valid_step(new_sent):
            classification_score, rank = get_classification_logprob(classification_model, input, prev_gen_text , new_sent, target_label) # prev_gen.text contains all previous generations
            logsum += prev_gen_logsum
            num_tokens_generated += prev_gen_num_tokens
            item = GenerationItem(curr_gen_ids, logsum, classification_score, text, num_tokens_generated, classification_rank = rank)
        else:
            item = prev_gen
            # classification_score, rank, logsum, num_tokens_generated = prev_gen.classification_score, prev_gen.classification_rank, prev_gen.logsum, prev_gen.num_tokens_generated
        

        generations.append(item)

    return generations
    
def generate_step(
    model,
    input_ids,
    gen_mode,
    stopping_criteria, 
    max_length, 
    top_p, 
    do_sample=True, 
    decoder_input_ids=None, 
    early_stopping=True, 
    num_return_sequences=1, 
    num_beams=1, 
    output_scores=True, 
    return_dict_in_generate=True,
    init_beam_scores = None,
):  

    if decoder_input_ids is not None:
        return model.generate(
            input_ids=input_ids, 
            max_length=max_length, 
            do_sample=do_sample,
            early_stopping=early_stopping,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            stopping_criteria=stopping_criteria,
            decoder_input_ids=decoder_input_ids,
            gen_mode = gen_mode, # pass this to customized generation kwargs
            init_beam_scores = init_beam_scores,
        )
    else: 
        outs = model.generate(
            input_ids=input_ids, 
            max_length=max_length, 
            do_sample=do_sample,
            early_stopping=early_stopping,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            stopping_criteria=stopping_criteria,
            gen_mode = gen_mode, # pass this to customized generation kwargs
            init_beam_scores = init_beam_scores,
        )
        return outs

def sort_filter_gen_history(sent_options:List[GenerationItem], n:int, alpha=1, beta=1): # n is the number of top sentences to select
    return sorted(sent_options, key=lambda item: (item.get_avg_log() * alpha + item.classification_score * beta), reverse=True)[:n] # sort in descending order

def sort_filter_gen_history_with_length_penalty(sent_options:List[GenerationItem], n:int): # n is the number of top steps to select
    return sorted(sent_options, key=lambda item: item.seq_score, reverse=True)[:n] # sort in descending order

def sort_filter_gen_histrory_by_rank(sent_options:List[GenerationItem], n:int):
    logsum_scores = [item.get_avg_log() for item in sent_options]
    logsum_sorted = sorted(logsum_scores, reverse=True)
    # print("sorted avg:", logsum_sorted)
    def get_combined_rank(item):
        avg = item.get_avg_log()
        if avg in logsum_sorted:
            logsum_rank = logsum_sorted.index(avg)
        else:
            print(colored(f"log sum avg not found in whole list: {avg} | {logsum_sorted}", 'red'))
   
        # logsum_rank = (logsum_sorted==item.get_avg_log()).nonzero()[0][0].item() # if multiple items have same score, take the earlier rank
        # logsum_rank = (logsum_sorted==item.get_avg_log()).nonzero().squeeze().item()
        comb_rank = logsum_rank + item.classification_rank
        return (comb_rank, item.classification_rank) # specify sorting secondary key
    sorted_items = sorted(sent_options, key=get_combined_rank, reverse=False) # the lower the score, the better the overall performance
    # for i, item in enumerate(sorted_items):
    #     print(i, ":", get_combined_rank(item), item.classification_rank, item.get_avg_log(), item.text)
    return sorted_items[:n]

def sort_filter_gen_history_with_classification_rank(sent_options:List[GenerationItem], n:int): # n is the number of top sentences to select
    # classification rank: the lower the better, reverse as compared to avg logsum
    return sorted(sent_options, key=lambda item: (item.get_avg_log()-item.classification_rank), reverse=True)[:n] # sort in descending order

def remove_duplicate_candidates(sent_options:List[GenerationItem]):
    visited = set()
    uniq_sents = []
    for item in sent_options:
        if item.text not in visited:
            visited.add(item.text)
            uniq_sents.append(item)

    return uniq_sents

