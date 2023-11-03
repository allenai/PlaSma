import json
from tqdm import tqdm
import logging
import argparse
import re
import os
import time
import math
from typing import TypeVar, Iterable, List, Union, Any
import torch
from typing import Tuple, List, Optional, Union
from termcolor import colored
import math 
import numpy as np
import re



from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    set_seed,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BeamSearchScorer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
)
from stopping_criteria import ( # self-defined
    EndSentenceCriteria,
    EndSpanCriteria,
    MultiBatchEndSentenceCriteria,
    MultiBatchEndStepCriteria
)
from stepwise_beam_utils import sort_filter_gen_history, generate_step, process_beamsearch_generation, process_multisample_generation, remove_duplicate_candidates
from beam_search_utils import (
    sample,
    beam_search,
    beam_sample,
)
from proto import GenerationItem

from generation_utils import GenerationMixin # import generate
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

labels2idx={
    "bad": 0,
    "good": 1,
}

def main():
    parser = argparse.ArgumentParser()
    # Other parameters

    parser.add_argument(
        "--model_path", type=str, default="faezeb/script-kd/k-distill/outputs/multitask/grouped_by_goal_v3/NEW_Feb23/t5-11b_bs8_lr1e-5_ep15_seed16797", help=""
    )
    parser.add_argument(
        "--classification_model_path", type=str, default="faezeb/script-kd/classifier/decoding_script_critic/outputs/v3/roberta-large_bs_32_ep_10_lr_1e-5_seed_3331", help=""
    ) # need to change / just for test
    # Optional
    parser.add_argument(
        "--max_length", default=180, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--max_source_length", default=60, type=int, required=False, help="Maximum text length/ 25 for script gen"
    )
    parser.add_argument(
        "--gen_size", default=10, type=int, required=False, help="number of step options to generate in first round"
    )
    parser.add_argument(
        "--gen_mode", type=str, help="mode of generations", choices=['beam_p_sampling', 'beam_k_sampling', 'beam_p_k', 'beam_sample', 'sample_p'], default="beam_p_sampling"
    )
    parser.add_argument(
        "--source_prefix", type=str, help="encoder source prefix", default="provide steps:"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0.9, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=5, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--alpha", default=1.0, type=float, required=False, help="alpha factor for lm contribution in overall score"
    )
    parser.add_argument(
        "--beta", default=1.0, type=float, required=False, help="beta factor for critic contribution in overall score"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default=0, type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--n_gpu", default=1, type=str, help="number of gpu."
    )
    parser.add_argument(
        "--task", type=str, help="task of generations", choices=['script', 'update', 'conditional', 'script-multi', 'update-multi', 'conditional-multi'], default="script"
    )
    parser.add_argument('--debug', action="store_true", default=False, help="Whether in debug mode")
    args = parser.parse_args()
    logger.debug(args)

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )


    # load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)

    # parallelize
    args.n_gpu = torch.cuda.device_count()
    if args.n_gpu > 1:
        device_ids = [0,1]
        device_map = {device_ids[0]: list(range(0, 12)),device_ids[1]: list(range(12,24))}
        model.parallelize(device_map)

    logger.info(f"Model Loaded from {args.model_path}")

    # NOTE: here using self-defined sample function to override what is defined in generation_utils
    model.sample = sample.__get__(model)

    # TODO: write customized function rather than replacing the original !!!
    if args.gen_mode == "beam_p_sampling": 
        model.beam_search = beam_search.__get__(model)

    model.tokenizer = tokenizer
    model.generate = GenerationMixin.generate.__get__(model)

    # load classifier (step-level)
    classification_model = pipeline('text-classification', model=args.classification_model_path, return_all_scores=True, device=args.n_gpu-1)


    while True:
        # get input data
        input_goal = input("Enter goal input (e.g., attend a medical school) or enter q to quit: ")
        if input_goal == 'q':
            break
        # condition - Implemented but need to merge this later
        condition = input("Enter condition input or just enter if no condition: ")
        input_formatted = format_data(args, input_goal, condition)

        # generate scripts
        output = generate_stepwise_beam(args, tokenizer, model, input_formatted, classification_model, device=device)
        logger.info(f"Generated script: {output.rstrip('Step').strip() if output.endswith('Step') else output.strip()}")


def generate_stepwise_beam(args, tokenizer, model, input, classification_model, device="cuda:0"):
    #TODO: avoid continuing candidates which are complete 
    torch.manual_seed(0)
    np.random.seed(0)
    if args.gen_mode == "beam_p_sampling":
        gen_history = []

        beamsearch_stopped = False # used to track if no need for beamsearch

        step = 1
        if args.debug:
            logger.info(f"\nGoal: {input}\n")

        # while not beamsearch_stopped:
        # Faeze added
        while gen_history == [] or not all([ex.beamsearch_stopped for ex in gen_history]):
            if args.debug:
                logger.info(f"\n\nstep: {step}")
            
            if step == 1:
                # The first step starts with Step 1: and we don't want to stop ("Step" is in our stoppingcritera)
                decoder_input_ids = torch.tensor([tokenizer.encode("<pad>Step 1:")[:-1]]).to(device)

                sent_options, beamsearch_stopped = generate_step_options(
                    args,
                    tokenizer, 
                    model, 
                    input,
                    classification_model,
                    target_label="good",
                    prev_beamsearch_stopped=beamsearch_stopped, 
                    decoder_input_ids = decoder_input_ids,
                    device=device)
                # newly added (March 9)
                sent_options = remove_duplicate_candidates(sent_options)
                gen_history = sort_filter_gen_history(sent_options, args.beams, args.alpha, args.beta) # get top k hypothesis

            else:
                sent_options = []
                for i, prev_item in enumerate(gen_history):
                    if args.debug:
                        print("\nprev state: logsum {} | num tokens {} | avg log {} | class prob {} | rank {} | overal score {} | {} ".format(prev_item.logsum,prev_item.num_tokens_generated, prev_item.get_avg_log(), prev_item.classification_score, prev_item.classification_rank, prev_item.get_avg_log() * args.alpha + prev_item.classification_score * args.beta, prev_item.text))
                    decoder_input_ids = None

                    batch_options, beamsearch_stopped = generate_step_options(
                        args,
                        tokenizer,
                        model,
                        input,
                        classification_model,
                        target_label="good",
                        prev_gen=prev_item, 
                        prev_beamsearch_stopped=beamsearch_stopped, 
                        decoder_input_ids = decoder_input_ids,
                        device=device)
                    if args.debug:
                        print("\nnew generations:")
                        # TODO: change overal scores depending on the aggregate function (sum, weighted sum, etc.)
                        for j, gen_item in enumerate(batch_options):
                            print("logsum {} | num tokens {} | avg log {} | class prob {} | rank {} | overal score {} | {}".format(gen_item.logsum,gen_item.num_tokens_generated,gen_item.get_avg_log(), gen_item.classification_score, gen_item.classification_rank, gen_item.get_avg_log() * args.alpha + gen_item.classification_score * args.beta, gen_item.text))
                    sent_options.extend(batch_options)


                sent_options = remove_duplicate_candidates(sent_options)
                gen_history = sort_filter_gen_history(sent_options, args.beams, args.alpha, args.beta)
                

            if args.debug:
                print("\nnew state:", len(gen_history), " --------------------------------------------------")
                for i, gen_item in enumerate(gen_history):
                    print("logsum {} | num tokens {} | avg log {} | class prob {} | rank {} | overal score {} | {}".format(gen_item.logsum,gen_item.num_tokens_generated,gen_item.get_avg_log(), gen_item.classification_score, gen_item.classification_rank, gen_item.get_avg_log() * args.alpha + gen_item.classification_score * args.beta, gen_item.text))
                print("\n\n")
            # import pdb; pdb.set_trace()
            step += 1
            

        output_text = sort_filter_gen_history(gen_history, 1, args.alpha, args.beta)[0].text
    else:
        raise NotImplementedError
    return output_text

def generate_step_options(
    args,
    tokenizer,
    model,
    input_goal: str, 
    clf_model,
    target_label: str,
    prev_gen: Optional[GenerationItem] = None,
    prev_beamsearch_stopped: Optional[bool] = False,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    device="cuda:0"
):
    """
        sample_size: number of sentences to generate from sampling
        num_sents: number of new sentences to generate
        input_ids: input_ids from source
        prev_gen: previously generated sentence class
        target_label: the idx for the intended generation
        decoder_input_ids: directly specify the decoder input ids if using ITSP, containing added prompt tokens
    """
    input_ids = tokenizer(input_goal, max_length=args.max_source_length, padding=False, truncation=True, return_tensors="pt").input_ids.to(device)
    beamsearch_stopped = False # flag this once beam search cannot generate anymore steps
    generations = []
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(EndSentenceCriteria(tokenizer=tokenizer))
    multibatch_stopping_criteria = StoppingCriteriaList()
    multibatch_stopping_criteria.append(MultiBatchEndStepCriteria(tokenizer))
    multibatch_stopping_criteria.append(MultiBatchEndSentenceCriteria(tokenizer.pad_token_id))
    decoder_input_ids= decoder_input_ids if decoder_input_ids is not None else prev_gen.token_ids if prev_gen is not None else None
    decoder_input_id_length = decoder_input_ids.size(1) if decoder_input_ids is not None else 0
    start_pos = decoder_input_ids.size(-1) if decoder_input_ids is not None else prev_gen.token_ids.size(-1) if prev_gen is not None else 1

    if decoder_input_id_length >= args.max_length: # no need to generate further if exceed max length
        item = prev_gen
        item.classification_score = 0
        generations.append(item)
        print(colored(f"generation force stopped due to exceeding max length, you may consider use longer MAX_TARGET_LENGTH", 'red'))
    
    # generation is a complete script - no need to generate further
    elif prev_gen is not None and not prev_gen.text.endswith("Step"):
        # import pdb; pdb.set_trace()
        item = prev_gen
        beamsearch_stopped = True
        item.beamsearch_stopped = beamsearch_stopped
        generations.append(item)

    else:
        if prev_gen is None or not prev_gen.beamsearch_stopped:
            # beam search
            beamsearch_outputs = generate_step(
                model,
                input_ids,
                args.gen_mode,
                multibatch_stopping_criteria,
                max_length=args.max_length,
                top_p=args.p,
                do_sample=False,
                num_beams= args.beams,
                decoder_input_ids=decoder_input_ids,
            )

            item, beamsearch_stopped = process_beamsearch_generation(tokenizer, input_goal, beamsearch_outputs, start_pos, clf_model, prev_gen = prev_gen, target_label=target_label, decoder_input_ids=decoder_input_ids)

            if not beamsearch_stopped:
                generations.append(item)
            # else:
            #     import pdb; pdb.set_trace()
        
        # neucleus sampling
        sample_outputs = generate_step(
            model,
            input_ids, 
            args.gen_mode,
            multibatch_stopping_criteria,
            max_length=args.max_length,
            top_p=args.p,
            num_beams=1,
            num_return_sequences=args.gen_size - len(generations),
            decoder_input_ids=decoder_input_ids,
        )
        
        items = process_multisample_generation(tokenizer, input_goal, sample_outputs, start_pos, clf_model, prev_gen = prev_gen, target_label=target_label, decoder_input_ids=decoder_input_ids)
        generations.extend(items)
    # import pdb; pdb.set_trace()
    return (generations, beamsearch_stopped)

def format_data(args, goal, condition=""):

    if args.task == 'script':
        return f"provide steps:{goal.strip('.')}" if "neg" not in args.model_path else f"{goal.strip('.')}"

    elif args.task == 'script-multi':
        return f"provide steps: [goal] {goal.strip('.')} [steps]"

    # elif args.task == 'update':
    #     return "goal": f"rewrite steps: [goal] {ex['goal'].strip('.')} [steps] {ex['script']} [condition] {ex['condition'].replace('If ','').strip('.')} [update]",
    #             "script": ' '.join([f"Step {i+1}: {step}" for i, step in enumerate(ex["updated_steps"])]),
    #             "source": ex["source"],
    #             "category": ex["category"],
    #             "prob": ex["prob"]                   
    #         }
    #         for ex in data
    #     ]

    elif args.task == 'conditional-multi':
        return f"provide steps conditionally: [goal] {goal.strip('.')} [condition] {re.sub('If |if ', '', condition).strip('.')} [steps]"

if __name__ == "__main__":
    main()

