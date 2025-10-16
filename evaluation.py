import torch
import os
import tqdm
import utils
import numpy as np
from evaluate import load
from transformers import BioGptTokenizer, BioGptForCausalLM
import argparse

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model_name, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    model = BioGptForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir='cache'
    ).bfloat16().cuda()

    tokenizer = BioGptTokenizer.from_pretrained(model_name)

    return model, tokenizer


def go(test_lists, model, tokenizer):
    invitation = "### Response: "
    human_invitation = "### Input: "
    generation_lists = []
    fulltexts = []

    for i in test_lists:
        history = ["Below is an instruction that describes a task, paired with an input that provides further context."
                   " Write a response that appropriately completes the request.\n\n### Instructions:\nIf you are a"
                   " doctor, please answer the medical questions based on the patient's description.",
                   human_invitation + '\n' + i]
        fulltext = "\n\n".join(history) + "\n\n" + invitation
        fulltexts.append(fulltext)

    for ft in tqdm.tqdm(fulltexts):
        gen_in = tokenizer(ft, return_tensors="pt").input_ids
        attn_masks = torch.ones_like(gen_in).cuda()
        gen_in = gen_in.cuda()
        with torch.no_grad():
            generated_ids = model.generate(
                gen_in,
                attention_mask=attn_masks,
                max_length=1024,
                use_cache=True,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=1.0,  # default: 1.0
                top_k=5,  # default: 50
                top_p=1.0,  # default: 1.0
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # for some reason, batch_decode returns an array of one element?
            text_without_prompt = generated_text[len(ft):]
            generation_lists.append(text_without_prompt)

    return generation_lists


def go_template_based(decoding, model, tokenizer):
    list_data_t = utils.jload('./datapreprocess/Template-ul.json')
    template = " The disease or symptom of {name} is"
    sources = [template.format_map(example) for example in list_data_t]
    targets = [s['symptom'] for s in list_data_t]

    total = len(targets)
    generation_lists = []

    invitation = "ChatDoctor: "
    human_invitation = "Patient: "

    for ft in tqdm.tqdm(sources):

        history = ["ChatDoctor: I am ChatDoctor, what medical questions do you have?", human_invitation + ft]
        ft = "If you are a doctor, please answer the medical questions based on the patient's description. " + " ".join(
            history) + " " + invitation
        
        gen_in = tokenizer(ft, return_tensors="pt")
        gen_in = {k: v.cuda() for k, v in gen_in.items()}

        if decoding == "greedy":
            with torch.no_grad():
                # topk and greedy decoding, only need to modify do_sample
                generated_ids = model.generate(
                    **gen_in,
                    max_new_tokens=256,
                    use_cache=True,
                    num_return_sequences=1,
                    do_sample=False,
                    repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                    temperature=1.0,  # default: 1.0
                    top_p=1.0,  # default: 1.0
                    early_stopping=True,
                )
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text_without_prompt = generated_text[len(ft):]
                generation_lists.append(text_without_prompt)

        elif decoding == "topk":
            generated_ids = model.generate(
                **gen_in,
                max_new_tokens=256,
                use_cache=True,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=1.0,  # default: 1.0
                top_k=5,  # default: 50
                top_p=1.0,  # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text_without_prompt = generated_text[len(ft):]
            generation_lists.append(text_without_prompt)

        elif decoding == "beam":
            generated_ids = model.generate(
               **gen_in,
               max_new_tokens=256,
               use_cache=True,
               num_return_sequences=5,  # Can be >1 to return top N beams
               num_beams=5,  # Number of beams (try 3–10 typically)
               early_stopping=True,
               repetition_penalty=1.1,  # Still useful
            )
            info = []
            for si in range(generated_ids.shape[0]):
                generated_text = tokenizer.batch_decode(generated_ids[si, :].unsqueeze(0), skip_special_tokens=True)[0]
                text_without_prompt = generated_text[len(ft):]
                info.append(text_without_prompt)
            info = ' '.join(info)
            generation_lists.append(info)

        else:
            raise ValueError("Please choose the right decoding approach")

    hit = 0
    for i in range(total):
        if targets[i] in generation_lists[i]:
            hit += 1
    percentage = hit / total

    return percentage


def calbertscore(generation, label, bert_model):
    bertscore = load('bertscore')
    predictions = generation
    references = label
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=4,
        lang="en"
    )
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluation of BERTscore or template-based query")
    parser.add_argument("--mode", choices=["BERTscore", "template-based query"], help="Choose the task")
    parser.add_argument("--bert_model", help="Choose the model to measure BERTscore")
    parser.add_argument("--model_path", help="Choose the path of weights")
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], help="Decoding strategies")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    if args.mode == "BERTscore":

        datas = utils.jload('iCliniq.json')
        input_lists = []
        reference_lists = []
        generations = []
        for e in datas:
            input_lists.append(e['input'])
            reference_lists.append(e['answer_icliniq'])
        generations = go(input_lists, model, tokenizer)
        bertscore = calbertscore(generations, reference_lists, args.bert_model)

        z = 1.96
        precision_m = np.average(bertscore['precision'])
        recall_m = np.average(bertscore['recall'])
        f1_m = np.average(bertscore['f1'])
        precision_s = np.std(bertscore['precision'], ddof=1) / np.sqrt(len(bertscore['precision']))
        recall_s = np.std(bertscore['recall'], ddof=1) / np.sqrt(len(bertscore['recall']))
        f1_s = np.std(bertscore['f1'], ddof=1) / np.sqrt(len(bertscore['f1']))
        precision_margin = z * precision_s
        recall_margin = z * recall_s
        f1_margin = z * f1_s

        print(f'precision: {precision_m}; recall: {recall_m}; f1: {f1_m}')
        print(f'precision_CI: {precision_margin}; recall_CI: {recall_margin}; f1_CI: {f1_margin}')

    elif args.mode == "template-based query":
        print(go_template_based(args.decoding, model, tokenizer))

    else:
        raise ValueError("Please choose the right mode")




