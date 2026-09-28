import gc
import torch
import numpy as np
import logging
import tqdm
import matplotlib.pyplot as plt
from GEP_utils import *
from utils import jload
import argparse


def generate(model, tokenizer, input_ids, assistant_role_slice, decoding, generation_length):
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    if decoding == "greedy":
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    max_new_tokens=generation_length,
                                    use_cache=True,
                                    num_return_sequences=1,
                                    do_sample=False,
                                    repetition_penalty=1.1,
                                    temperature=1.0,
                                    top_p=1.0,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        return output_ids[assistant_role_slice.stop:]

    elif decoding == "topk":
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    max_new_tokens=generation_length,
                                    use_cache=True,
                                    num_return_sequences=1,
                                    do_sample=True,
                                    repetition_penalty=1.1,
                                    temperature=1.0,
                                    top_p=1.0,
                                    top_k=5,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        return output_ids[assistant_role_slice.stop:]

    elif decoding == "beam":
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    max_new_tokens=generation_length,
                                    use_cache=True,
                                    num_return_sequences=5,
                                    num_beams=5,
                                    early_stopping=True,
                                    repetition_penalty=1.1)
        return [ids[assistant_role_slice.stop:] for ids in output_ids]

    else:
        raise ValueError("Please choose the right decoding approach")


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, decoding,
                             generation_length):
    jailbroken = False
    appear_idx = 0

    gen_str_ = generate(model,
                        tokenizer,
                        input_ids,
                        assistant_role_slice,
                        decoding,
                        generation_length)

    if decoding == "greedy" or decoding == "topk":

        gen_str = tokenizer.decode(gen_str_).strip()
        jailbroken = test_prefixes[0] in gen_str
        if jailbroken:
            generate_tokens = gen_str_.cpu().numpy().tolist()
            wanted_tokens = tokenizer(test_prefixes[0]).input_ids[1:]
            appear_idx = token_check(generate_tokens, wanted_tokens)
            logger.info(f'Generation is ###{gen_str}###, and test prefix is ###{test_prefixes[0]}###, '
                        f'and appear_idx is ###{appear_idx}###')
        else:
            appear_idx = -1

    elif decoding == "beam":

        for item in gen_str_:
            gen_str = tokenizer.decode(item).strip()
            jailbroken = test_prefixes[0] in gen_str
            if jailbroken:
                generate_tokens = item.cpu().numpy().tolist()
                wanted_tokens = tokenizer(test_prefixes[0]).input_ids[1:]
                appear_idx = token_check(generate_tokens, wanted_tokens)
                logger.info(f'Generation is ###{gen_str}###, and test prefix is ###{test_prefixes[0]}###, and '
                            f'appear_idx is ###{appear_idx}###')
                break
            else:
                appear_idx = -1

    else:
        raise ValueError("Please choose the right decoding approach")

    return jailbroken, appear_idx


def token_check(generation, wanted):
    sub_len = len(wanted)
    for a in range(len(generation) - sub_len + 1):
        if generation[a: a + sub_len] == wanted:
            return a
    return -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GEP for template-based PII insertion")

    parser.add_argument("--model_path", help="Choose the path of weights")
    parser.add_argument("--model_type", choices=["biogpt", "opt", "qwen3"])
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], help="Decoding strategies")
    parser.add_argument("--trigger_length", type=int, help="The length of the trigger tokens")
    parser.add_argument("--steps", type=int, help="The total steps/iteration of the training")
    parser.add_argument("--bs", type=int, help="The batch size for the replacement of each trigger token")
    parser.add_argument("--topk", type=int, help="The top-k candidates to choose from for the replacement")
    parser.add_argument("--generation_length", type=int, help="The maximum tokens in the generation")
    parser.add_argument("--loss_curve", type=bool, help="To draw the loss curves for each entry")

    args = parser.parse_args()

    device = torch.device('cuda')

    model, tokenizer = load_model_and_tokenizer(args.model_path,
                                                args.model_type,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(f'./logs/attackprompt_single_{args.model_type}.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    adv_string_init = " ".join("!" for _ in range(args.trigger_length))
    target = "disease or symptom is"
    allow_non_ascii = False
    acc = 0
    asr_step = np.zeros(args.steps, dtype=int)
    asr_generation = np.zeros(args.generation_length, dtype=int)

    datadict = jload('./datapreprocess/Template-ul.json')
    for _n in tqdm.tqdm(datadict):

        user_prompt = _n['name']
        test_prefixes = [_n['symptom']]
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                                       instruction=user_prompt,
                                       target=target,
                                       adv_string=adv_string_init,
                                       model_type=args.model_type)

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        adv_suffix = adv_string_init
        step = 0
        losses_ = []

        for i in range(args.steps):

            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(device)

            coordinate_grad = token_gradients(model,
                                              input_ids,
                                              suffix_manager._control_slice,
                                              suffix_manager._target_slice,
                                              suffix_manager._loss_slice,
                                              args.model_type)

            with torch.no_grad():

                adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

                # Sample the new candidate trigger tokens
                new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                                     coordinate_grad,
                                                     args.bs,
                                                     topk=args.topk,
                                                     temp=1,
                                                     not_allowed_tokens=not_allowed_tokens,
                                                     vs=tokenizer.vocab_size)

                new_adv_suffix = get_filtered_cands(tokenizer,
                                                    new_adv_suffix_toks,
                                                    args.trigger_length,
                                                    filter_cand=True,
                                                    curr_control=adv_suffix)

                # calculate the loss for each candidate
                logits, ids = get_logits(model=model,
                                         tokenizer=tokenizer,
                                         input_ids=input_ids,
                                         control_slice=suffix_manager._control_slice,
                                         test_controls=new_adv_suffix,
                                         return_ids=True,
                                         batch_size=args.bs)

                losses = target_loss(logits, ids, suffix_manager._target_slice)

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id].lstrip()

                current_loss = losses[best_new_adv_suffix_id]
                losses_.append(current_loss.detach().cpu().numpy())

                step = step + 1

                # Updating the best trigger tokens
                adv_suffix = best_new_adv_suffix
                is_success, idx = check_for_attack_success(model,
                                                           tokenizer,
                                                           suffix_manager.get_input_ids(adv_string=adv_suffix).to(
                                                               device),
                                                           suffix_manager._assistant_role_slice,
                                                           test_prefixes,
                                                           args.decoding,
                                                           args.generation_length)

            if is_success:
                print(f"\nPassed:{is_success}\nCurrent Suffix:{best_new_adv_suffix}", end='\r')
                logger.info(f'{user_prompt}\'s attack trigger is: {best_new_adv_suffix}')
                acc = acc + 1
                asr_step[i] = asr_step[i] + 1
                if idx > -1:
                    asr_generation[idx] = asr_generation[idx] + 1
                break

            # (Optional) Clean up the cache.
            del coordinate_grad, adv_suffix_tokens
            gc.collect()
            torch.cuda.empty_cache()

        if args.loss_curve:
            plt.figure()
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.plot(np.arange(len(losses_)), losses_)
            plt.savefig(f'./GEPloss/loss_{user_prompt}.png')
            plt.close()

    acc_rate = acc / 1000
    print(f'Successful attack\'s rate is {acc_rate}')
    logger.info(f'Successful attack\'s rate is {acc_rate}')
    logger.info(f'The number of successful attacks at each step is: {asr_step}')
    logger.info(f'The number of successful attacks at each index in the final generation is: {asr_generation}')
