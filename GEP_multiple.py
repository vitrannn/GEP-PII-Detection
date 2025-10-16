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

        return output_ids[assistant_role_slice.stop:], tokenizer.decode(input_ids[0][1:])

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
        return output_ids[assistant_role_slice.stop:], tokenizer.decode(input_ids[0][1:])

    elif decoding == "beam":
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    max_new_tokens=generation_length,
                                    use_cache=True,
                                    num_return_sequences=5,
                                    num_beams=5,
                                    early_stopping=True,
                                    repetition_penalty=1.1)
        return [ids[assistant_role_slice.stop:] for ids in output_ids], tokenizer.decode(input_ids[0][1:])

    else:
        raise ValueError("Please choose the right decoding approach")


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, decoding, generation_length):
    jailbroken = False
    appear_idx = 0

    gen_str_, input_str = generate(model,
                                   tokenizer,
                                   input_ids,
                                   assistant_role_slice,
                                   decoding,
                                   generation_length)

    if decoding == "greedy" or decoding == "topk":

        gen_str = tokenizer.decode(gen_str_).strip()
        jailbroken = test_prefixes in gen_str
        if jailbroken:
            generate_tokens = gen_str_.cpu().numpy().tolist()
            wanted_tokens = tokenizer(test_prefixes).input_ids[1:]
            appear_idx = token_check(generate_tokens, wanted_tokens)
            logger.info(f'Input is ###{input_str}###, and generation is ###{gen_str}###, and test prefix is '
                        f'###{test_prefixes}###, and appear_idx is ###{appear_idx}###')
        else:
            appear_idx = -1

    elif decoding == "beam":

        for item in gen_str_:
            gen_str = tokenizer.decode(item).strip()
            jailbroken = test_prefixes in gen_str
            if jailbroken:
                generate_tokens = item.cpu().numpy().tolist()
                wanted_tokens = tokenizer(test_prefixes[0]).input_ids[1:]
                appear_idx = token_check(generate_tokens, wanted_tokens)
                logger.info(f'Input is ###{input_str}###, and generation is ###{gen_str}###, and test prefix is'
                            f' ###{test_prefixes[0]}###, and appear_idx is ###{appear_idx}###')
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


def data_split(ds, r, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ds = np.array(ds)
    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    tr_data_len = int(len(ds) * r)
    tr_idx = indices[:tr_data_len]
    va_idx = indices[tr_data_len:]
    return ds[tr_idx].tolist(), ds[va_idx].tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GEP for free-style PII insertion")

    parser.add_argument("--model_path", help="Choose the path of weights")
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], help="Decoding strategies")
    parser.add_argument("--trigger_length", type=int, help="The length of the trigger tokens")
    parser.add_argument("--steps", type=int, help="The total steps/iteration of the training")
    parser.add_argument("--bs", type=int, help="The batch size for the replacement of each trigger token")
    parser.add_argument("--topk", type=int, help="The top-k candidates to choose from for the replacement")
    parser.add_argument("--generation_length", type=int, help="The maximum tokens in the generation")
    parser.add_argument("--loss_curve", type=bool, help="To draw the loss curves for all entries")
    parser.add_argument("--split_ratio", type=float, help="The ratio to split the training and validation set")

    args = parser.parse_args()

    device = torch.device('cuda')

    model, tokenizer = load_model_and_tokenizer(args.model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler('./logs/attackprompt_multiple.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    adv_string_init = " ".join("!" for _ in range(args.trigger_length))
    allow_non_ascii = False

    datadict = jload('./datapreprocess/Template-ul.json')
    tr_dict, va_dict = data_split(datadict, args.split_ratio)
    tr_suffix_ms = []  # save the SuffixManager class for each entry in the training set
    va_suffix_ms = []  # save the SuffixManager class for each entry in the validation set
    tr_names = []  # save the patient's name in the training set
    va_names = []  # save the patient's name in the validation set
    tr_targets = []  # save the generation target, i.e., disease for each patient in the training set
    va_targets = []  # save the generation target, i.e., disease for each patient in the validation set
    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
    adv_suffix = adv_string_init
    losses_ = []  # save the total loss for each step
    asr_step_tr = []
    asr_step_va = []

    # Record the meta data for each entry in training and validation set
    for _n in tr_dict:
        user_prompt = _n['name']
        target = _n['symptom']
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                                       instruction=user_prompt,
                                       target=target,
                                       adv_string=adv_string_init)
        tr_suffix_ms.append(suffix_manager)
        tr_names.append(user_prompt)
        tr_targets.append(target)

    for _n in va_dict:
        user_prompt = _n['name']
        target = _n['symptom']
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                                       instruction=user_prompt,
                                       target=target,
                                       adv_string=adv_string_init)
        va_suffix_ms.append(suffix_manager)
        va_names.append(user_prompt)
        va_targets.append(target)

    for i in tqdm.tqdm(range(args.steps)):

        tr_lss = []
        va_lss = []
        tr_coordinate_grads = []
        tr_input_ids = []
        for ts in tr_suffix_ms:
            input_ids = ts.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(device)
            tr_input_ids.append(input_ids)

            # calculate the gradient of the trigger tokens
            coordinate_grad_ = token_gradients(model,
                                               input_ids,
                                               ts._control_slice,
                                               ts._target_slice,
                                               ts._loss_slice)
            tr_coordinate_grads.append(coordinate_grad_)

        coordinate_grad = sum(tr_coordinate_grads) / len(tr_dict)

        with torch.no_grad():

            adv_suffix_tokens = torch.tensor(tokenizer(adv_suffix).input_ids[1:], dtype=torch.int64).to(device)

            # sample the new candidate trigger tokens
            new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                                 coordinate_grad,
                                                 args.bs,
                                                 topk=args.topk,
                                                 temp=1,
                                                 not_allowed_tokens=not_allowed_tokens)

            new_adv_suffix = get_filtered_cands(tokenizer,
                                                new_adv_suffix_toks,
                                                filter_cand=True,
                                                curr_control=adv_suffix)

            # calculate the total loss on all the entries
            for j in range(len(tr_input_ids)):
                input_ids = tr_input_ids[j]
                ts = tr_suffix_ms[j]
                logit, ids = get_logits(model=model,
                                        tokenizer=tokenizer,
                                        input_ids=input_ids,
                                        control_slice=ts._control_slice,
                                        test_controls=new_adv_suffix,
                                        return_ids=True,
                                        batch_size=args.bs)  # decrease this number if you run into OOM.
                loss = target_loss(logit, ids, ts._target_slice)
                tr_lss.append(loss)

            losses = sum(tr_lss) / len(tr_dict)
            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id].detach().cpu().numpy()
            losses_.append(current_loss)

            # Update the trigger tokens by the loss, and check on the val set
            adv_suffix = best_new_adv_suffix

            logger.info(f'\n'
                        f'=====================================================================\n'
                        f'============================Step{i}==================================\n'
                        f'=====================================================================')
            logger.info(f'Current triggers are {adv_suffix}')
            logger.info(f'\n'
                        f'===============================================\n'
                        f'===================Train=======================\n'
                        f'===============================================')

            tr_count = 0
            asr_generation_tr = np.zeros(256, dtype=int)
            for k in range(len(tr_dict)):
                is_success, idx = check_for_attack_success(model,
                                                           tokenizer,
                                                           tr_suffix_ms[k].get_input_ids(adv_string=adv_suffix).to(device),
                                                           tr_suffix_ms[k]._assistant_role_slice,
                                                           tr_targets[k],
                                                           args.decoding,
                                                           args.generation_length)
                tr_count = tr_count + is_success
                if is_success:
                    if idx > -1:
                        asr_generation_tr[idx] = asr_generation_tr[idx] + 1

            tr_acc = tr_count / len(tr_dict)
            asr_step_tr.append(tr_acc)
            logger.info(f'ASR w.r.t train generation is: {asr_generation_tr}')

            logger.info(f'\n'
                        f'====================================================\n'
                        f'===================Validation=======================\n'
                        f'====================================================')

            va_count = 0
            asr_generation_va = np.zeros(256, dtype=int)
            for m in range(len(va_dict)):
                is_success, idx = check_for_attack_success(model,
                                                           tokenizer,
                                                           va_suffix_ms[m].get_input_ids(adv_string=adv_suffix).to(device),
                                                           va_suffix_ms[m]._assistant_role_slice,
                                                           va_targets[m],
                                                           args.decoding,
                                                           args.generation_length)
                va_count = va_count + is_success
                if is_success:
                    if idx > -1:
                        asr_generation_va[idx] = asr_generation_va[idx] + 1

            va_acc = va_count / len(va_dict)
            asr_step_va.append(va_acc)
            logger.info(f'The number of successful attacks at each index in the final generation is: {asr_generation_va}')

            logger.info(f'On iteration of {i}, training set\'s ASR is {tr_acc}, validation set\'s ASR is {va_acc}, '
                        f'loss is {current_loss}')

        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    if args.loss_curve:
        plt.figure()
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.plot(np.arange(args.steps), losses_)
        plt.savefig(f'./GEPloss/loss.png')
        plt.close()

    logger.info('============================')
    tr_accs = np.array(asr_step_tr)
    va_accs = np.array(asr_step_va)
    logger.info(f'The ASR at each training step is: {asr_step_tr}')
    logger.info(f'The ASR at each validation step is: {asr_step_va}')
    best_ep_tr = np.argmax(tr_accs)
    best_ep_va = np.argmax(va_accs)
    logger.info(f'The best train trigger happens in iteration {best_ep_tr}, where the ASR is {tr_accs[best_ep_tr]}')
    print(f'The best train trigger happens in iteration {best_ep_tr}, where the ASR is {tr_accs[best_ep_tr]}')
    logger.info(f'The best validation trigger happens in iteration {best_ep_va}, where the ASR is {va_accs[best_ep_va]}')
    print(f'The best validation trigger happens in iteration {best_ep_va}, where the ASR is {va_accs[best_ep_va]}')
