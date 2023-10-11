import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchOutput,
    GreedySearchOutput,
    SampleOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
)

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
# from transformers.modeling_utils import PreTrainedModel

from datasets import load_metric
import numpy as np
import re
import pandas as pd
from torch import nn
# from torch.utils.data import DataLoader, Dataset

def postprocess_text(preds, golds):
    preds = [pred.strip() for pred in preds]
    golds = [label.strip() for label in golds]

    return preds, golds

def compute_metrics(preds, golds):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(preds, golds)
    metric = load_metric("./rouge")
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}
    return result

def get_prompts_from_input_text(text, pre_prompt, post_prompt):
    prompt_string = text[:text.index(post_prompt)]
    prompts = prompt_string.strip().split(pre_prompt)
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()!=""]
    return prompts

def remove_prompts(input_str, rm_type="extra_tokens"):
    """
        remove all labels by identifying the patterns
    """
    if "special_sep" in rm_type:
        prompts = re.findall(r'\¥\s[^\s]+\s\þ?',input_str)
    elif "extra_tokens" in rm_type:
        # prompts = re.findall(r'<label-sep>\s?[a-z_A-Z0-9]+\s?<sent-sep>|<label-sep>\s?[a-z_A-Z0-9]+\s?|\s?[a-z_A-Z0-9]+\s?<sent-sep>', input_str)
        prompts = re.findall(r'<label-sep>[a-z_A-Z]+<sent-sep>', input_str)

    else: # pattern "| label1 ==>" needs to be removed
        prompts = re.findall(r'\|\s[^\s]+\s=*>?',input_str)

    for prompt in prompts:
        start = input_str.find(prompt)
        end = start + len(prompt)
        if "extra_token" in rm_type:
            input_str = input_str[:start]+' '+input_str[end:]
        else:
            input_str = input_str[:start]+input_str[end+1:]

    if input_str.strip() == "": # DEBUG
        print("got empty!")

    return input_str

def get_logits_processor(
        config,
        # repetition_penalty: float,
        # no_repeat_ngram_size: int,
        # encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        # bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        # eos_token_id: int,
        # forced_bos_token_id: int,
        # forced_eos_token_id: int,
        num_beams: int,
        # num_beam_groups: int,
        # diversity_penalty: float,
        # remove_invalid_values: bool,
):  
    processors = LogitsProcessorList()
    repetition_penalty = config.repetition_penalty
    no_repeat_ngram_size = (config.no_repeat_ngram_size)
    encoder_no_repeat_ngram_size = (config.encoder_no_repeat_ngram_size)
    # encoder_input_ids = encoder_input_ids
    bad_words_ids = config.bad_words_ids
    # min_length = min_length
    # max_length = max_length
    eos_token_id = config.eos_token_id
    diversity_penalty = config.diversity_penalty
    # num_beams = num_beams
    num_beam_groups = config.num_beam_groups
    forced_bos_token_id = (config.forced_bos_token_id)
    forced_eos_token_id = (config.forced_eos_token_id)
    remove_invalid_values = (config.remove_invalid_values)

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    # default no_repeat_ngram_size is 3
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0: 
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))

    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    # we use 1 for min_length
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    # if prefix_allowed_tokens_fn is not None:
    #     processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    # default forced_bos_token_id is 0
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    # default forced_eos_token_id is 2
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors

def beam_search_sent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    input_ids: torch.LongTensor,
    prompt_ids: List[Tuple[List[int], str]],
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    assert (
        num_beams * batch_size == batch_beam_size
    ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = next_token_logits
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        
        # SCH DEBUG: 
        # print("input id:\n",input_ids)
        sent_terminator_id = tokenizer.convert_tokens_to_ids("<label-sep>")
        label_terminator_id = tokenizer.convert_tokens_to_ids("<sent-sep>")
        for i in range(len(input_ids)):
            curr_gen = tokenizer.decode(input_ids[i][-50:-1], skip_special_tokens=False, clean_up_tokenization_spaces=True).strip()
            # NOTE: use -50: to speed up, and -1 because we need to set last token to <label-sep> if prev sentence ended already
            # print("see if need to cut:", input_ids[i][:-1])
            if is_sent_complete(curr_gen):
                input_ids[i][-1] = sent_terminator_id
                continue # done processing

            input_list = input_ids[i].tolist()
            curr_prompt_pos = input_list[:-1].count(sent_terminator_id) - 2 # we don't handle anything if <label-sep> is just added or generated
            if curr_prompt_pos < 0:
                continue # no need processing

            # if curr_prompt_pos > len(prompt_ids):
            #     print("error: generated too many labels:", curr_prompt_pos, len(prompt_ids), curr_gen)
            #     print([y for x,y in prompt_ids])
            #     # print("ids:", input_list)
            #     continue
            if len(prompt_ids) == 0:
                continue
            if curr_prompt_pos > len(prompt_ids):
                # print("generated more <label-sep> than needed")
                continue # nothing we can do
            elif curr_prompt_pos == len(prompt_ids): # see if misgenerated consequtive ones
                if input_list[-len(prompt_ids[-1][0]):-1].count(sent_terminator_id)> 1: 
                    curr_prompt_pos -= 1
                    curr_prompt_id, curr_prompt = prompt_ids[-1]
                    curr_prompt_len = len(curr_prompt_id)
                    # print("misgenerated <label-sep> ignored")
                else:
                    continue # nothing we can do, but we can cut the generation later 
            else:    
                curr_prompt_id, curr_prompt = prompt_ids[curr_prompt_pos]
                # avoid: a new <label-sep> is generated before previous label prompt is completed
                
                prev_prompt_len = None
                if curr_prompt_pos > 0:
                    prev_prompt_id, _ = prompt_ids[curr_prompt_pos-1]
                    prev_prompt_len = len(prev_prompt_id)
                    
                curr_prompt_len = len(curr_prompt_id)
                traceback_length = curr_prompt_len + prev_prompt_len if prev_prompt_len else curr_prompt_len 
                # NOTE: previous prompt hasn't finished and we are one prompt ahead
                if input_list[-traceback_length:-1].count(sent_terminator_id)> 1 :
                    if curr_prompt_pos > 0:
                        curr_prompt_id, curr_prompt = prompt_ids[curr_prompt_pos-1]
                        curr_prompt_len = len(curr_prompt_id)
                        curr_prompt_pos -= 1
                        # print("need to cleanup missgenerated <label-sep>:", curr_gen)
                        # print("traceback length:", traceback_length)
                        # print("current ids:", input_list)
                    else:
                        print("strange condition, no prev prompt but we are having two <label-sep> at the end: ", input_list)
                        continue # don't do anything yet
            
            curr_prompt_start_idx = [i for i, n in enumerate(input_list[:-1]) if n == sent_terminator_id][curr_prompt_pos+1] # +1 for adding to the first prompt

            if not label_terminator_id in input_list[curr_prompt_start_idx:-1]:
                ## NOTE: not label_terminator_id prevents appending curr_prompt directly aft prev_prompt when no sentence has been generated
                ## should not check the last id in case system mistakenly generates a label_terminator_id, eg. <label-sep>rebuttal<sent-sep>, leading to not completing <label-sep>rebuttal_process<sent-sep>
                # print("need to add new prompt to: ", input_list)
                # print("curr prompt ids:", curr_prompt_id)
                # print("cutting pos:", curr_prompt_start_idx)
                # print("curr prompt:", curr_prompt)

                # pos = input_list[::-1].index(sent_terminator_id)
                # NOTE: ignore the sent_terminator_id at [-1] position
                pos = len(input_list)-curr_prompt_start_idx-1
                # print("pos:", pos)
                if pos >= len(curr_prompt_id):
                    print("error, pos exceed current prompt:", pos, curr_prompt_id, input_list)
                    continue
                input_ids[i][-1] =  curr_prompt_id[pos]
                # print("processed prompt:", input_ids[i])

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):     
            break
  
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    # DEBUG
    sequence_list = sequence_outputs["sequences"].tolist()[0]
    used_len = sequence_list.count(sent_terminator_id)-1
    if used_len < len(prompt_ids):
        unused = [x[-1] for x in prompt_ids[used_len:]]
        print("not using full label sequence, unsued labels are:", unused)
    if used_len > len(prompt_ids):
        # need to cut
        cutting_pos = [i for i, n in enumerate(sequence_list) if n == sent_terminator_id][len(prompt_ids)+1]
        print("force removing ", str(len(sequence_list)-cutting_pos), " tokens")
        sequence_outputs["sequences"] = sequence_outputs["sequences"][:,:cutting_pos] 

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
    else:
        return sequence_outputs["sequences"]


def beam_search(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    assert (
        num_beams * batch_size == batch_beam_size
    ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = next_token_logits
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):     
            break
  
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
    else:
        return sequence_outputs["sequences"]


def prepare_inputs_for_generation(
    decoder_input_ids,
    prompt_length=-1,
    past=None,
    attention_mask=None,
    head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs
):
    # cut decoder_input_ids if past is used
    if past is not None:
        if prompt_length == -1:
            decoder_input_ids = decoder_input_ids[:, -1:]
        else:
            decoder_input_ids = decoder_input_ids[:, -1-prompt_length:]

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
        "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    }


def is_sent_complete(text):
    # except: et al., p.s., e.g., i.e., aka., etc.,  w.r.t. vs., Figure C.1, James W., http(s)://arxiv.org/pdf/123.0003
    # don't exclude 1.2 yet: "[^,]\s[0-9]+\.",
    # don't include an unclosed bracket
    # if len(text) < 10: # enforce a sentence must be > 10 tokens
    #     return False 
    exception_indicators=["\se\.","\se\.?\s?g\.\s?","\sE\.","E\.?\s?g\.\s?","\set al\.\)?","\si\.", "\si\.?\s?e\.\s?","\sw\.","\sw\.?\s?r\.", "\sw\.?r\.?t.","\sa\.","\sa\.?\s?k\.", "\sa\.?k\.?a\.", \
        "\setc\.,","\sv\.","\sv\.?s\.", "\sp\.","\sp\.?s\.", "\s[A-Z][a-z]+\s[A-Z]\.", \
        "https?:[0-9\/a-zA-Z\.\-^\s]+\.", \
        "Con:? <sep> -", "Pro:? <sep> -","\([^\)]+\."]
    terminators = re.findall("|".join(exception_indicators), text)
    for item in terminators:
        if item == text[-len(item):]:
            return False
        
    end_indicators=["\;","\.\"?\s?\)?",'\?"?', '\!"?', "meta score: [0-9]", "Dear authors,"]
    terminators = re.findall("|".join(end_indicators), text)
    for item in terminators:
        if item == text[-len(item):]:
            return True

    return False

def is_step_complete(text):
    end_indicators=["Step"]
    terminators = re.findall("|".join(end_indicators), text)
    for item in terminators:
        if item == text[-len(item):]:
            return True

    return False

def greedy_search(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

    while True:
        # prepare model inputs
        # model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                )
                cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                )

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=True
        )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        curr_gen = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0].strip()
        
        model_inputs = prepare_inputs_for_generation(input_ids, prompt_length=-1, **model_kwargs) 

        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break

    return input_ids

def greedy_search_sent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    input_ids: torch.LongTensor,
    prompt_ids: List[Tuple[List[int], str]],
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    pad_token_id: int,
    eos_token_id: int,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

    prompt_pos = input_ids.size(1) # for preventing truncation of super short sentences
    while True:
        # prepare model inputs
        # model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                )
                cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                )

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=True
        )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        curr_gen = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0].strip()

        prompt_length = -1 # SCH: if not -1, used to notify inputs_preparation not to truncate input_ids

        # NOTE SCH: TODO use better ways to determine if a new sentence is completed
        cut_sent = is_sent_complete(curr_gen)
        # if cut_sent is True:
            # print("ending sent:", tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0].strip().encode("utf-8"))
        if input_ids.size(1) - prompt_pos > 0 and cut_sent is True: 
            if len(prompt_ids) > 0:
                # print("adding:", prompt_ids[0][-1])
                prompt_length = len(prompt_ids[0][-1])
                prompt_tensor = torch.tensor([prompt_ids[0][0]], device=input_ids.device).long()
                input_ids = torch.cat((input_ids, prompt_tensor), dim=1)
                assert input_ids.dim() == 2 and input_ids.size(0) == 1
                # print("currently:", tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0].strip().encode("utf-8"))
                prompt_pos = input_ids.size(1)
                prompt_ids = prompt_ids[1:]

        # prevent termination if prompt remaining
        if int(input_ids[:, -1].item()) == eos_token_id and len(prompt_ids)>0:
            # print("rmove </s> and adding:", prompt_ids[0][-1])
            prompt_length = len(prompt_ids[0][-1])
            prompt_tensor = torch.tensor([prompt_ids[0][0]], device=input_ids.device).long()
            input_ids = torch.cat((input_ids[:, :-1], prompt_tensor), dim=1)
            assert input_ids.dim() == 2 and input_ids.size(0) == 1
            prompt_pos = input_ids.size(1)
            prompt_ids = prompt_ids[1:]
        
        model_inputs = prepare_inputs_for_generation(input_ids, prompt_length, **model_kwargs) # NOTE: SCH: pass in prompt_length to let decoder_input_id not to ignore the prompt

        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break

    return input_ids



# def clean_input_for_critic(input):
#     for span in ["provide steps: ", "[goal]", "[steps]", "[update]"]:
#         input = input.replace(span, "")

#     return input.lower().strip().replace("  ", " ")

def clean_input_for_critic(input):
    '''
    task specific regex to extract ONLY the goal from the templated input
    '''
    update_task_start = "You want to "
    update_task_end = ". You need to "

    condition_task_end = ". How do you do this "

    other_tasks_start = "\[goal\]"
    other_tasks_end1 = "\[steps\]"
    other_tasks_end2 = "\[condition\]"

    regex = f'(?:(?<={update_task_start}).+?(?={update_task_end}|{condition_task_end}))|(?:(?<={other_tasks_start}).+?(?={other_tasks_end1}|{other_tasks_end2}))'

    match = re.findall(regex, input)

    if len(match) > 0:
        return match[0].lower().strip()

    return input.replace("provide steps:", "").lower().strip().replace("  ", " ")

def is_valid_step(candidate_step):
    if candidate_step in ['', "Step", "step", ".", ",", "!"]:
        return False
    return True