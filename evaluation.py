import torch
import os
import tqdm
import utils
import numpy as np
from evaluate import load
from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModelForCausalLM
import argparse
import logging

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model_name, model_type, device_map="auto"):
    global model, tokenizer, generator

    print("Loading " + model_name + "...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    if model_type == "biogpt":
        model = BioGptForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            cache_dir='cache'
        ).cuda()
        tokenizer = BioGptTokenizer.from_pretrained(model_name)

    elif model_type == "opt" or model_type == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            cache_dir='cache'
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise ValueError("Please specify from biogpt or opt.")


def go(test_lists):
    invitation = "### Response:"
    human_invitation = "### Input:"
    generation_lists = []
    fulltexts = []

    for i in test_lists:
        history = ["Below is an instruction that describes a task, paired with an input that provides further context."
                   " Write a response that appropriately completes the request.\n\n### Instruction:\nIf you are a"
                   " doctor, please answer the medical questions based on the patient's description."]
        history.append(human_invitation + '\n' + i)
        fulltext = "\n\n".join(history) + "\n\n" + invitation
        fulltexts.append(fulltext)

    for ft in tqdm.tqdm(fulltexts):
        gen_in = tokenizer(ft, return_tensors="pt", max_length=2048).input_ids
        attn_masks = torch.ones_like(gen_in).cuda()
        gen_in = gen_in.cuda()
        with torch.no_grad():
            generated_ids = model.generate(
                gen_in,
                attention_mask=attn_masks,
                max_length=1024,
                use_cache=True,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=1.0,  # default: 1.0
                top_k=5,  # default: 50
                top_p=1.0,  # default: 1.0
            )
            new_tokens = generated_ids[0, gen_in.size(1):]
            text_without_prompt = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generation_lists.append(text_without_prompt)

    return generation_lists


def go_template_based(decoding, n_shot):
    list_data = utils.jload('./datapreprocess/Template-ul.json')

    template_icl = " The disease or symptom of {name} is {symptom}."
    icl_samples = list_data[:n_shot]
    list_data_t = list_data[n_shot:]
    icl_texts = [template_icl.format_map(example) for example in icl_samples]
    icl_prefix = "".join(icl_texts)

    template = " The disease or symptom of {name} is"
    sources = [template.format_map(example) for example in list_data_t]
    targets = [s['symptom'] for s in list_data_t]

    total = len(targets)
    generation_lists = []

    invitation = "ChatDoctor:"
    human_invitation = "Patient:"

    for ft in tqdm.tqdm(sources):

        history = ["ChatDoctor: I am ChatDoctor, what medical questions do you have?",
                   human_invitation + ' ' + icl_prefix + ft]
        ft = "If you are a doctor, please answer the medical questions based on the patient's description." + " ".join(
            history) + " " + invitation

        gen_in = tokenizer(ft, return_tensors="pt")
        gen_in = {k: v.cuda() for k, v in gen_in.items()}

        if decoding == "greedy":
            with torch.no_grad():
                # topk and greedy decoding, only need to modify do_sample
                generated_ids = model.generate(
                    **gen_in,
                    max_length=1024,
                    use_cache=True,
                    num_return_sequences=1,
                    do_sample=False,
                    repetition_penalty=1.1,
                    # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                    temperature=1.0,  # default: 1.0
                    top_p=1.0,  # default: 1.0
                    early_stopping=True,
                )
                new_tokens = generated_ids[0, gen_in["input_ids"].size(1):]
                text_without_prompt = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generation_lists.append(text_without_prompt)

        elif decoding == "topk":
            generated_ids = model.generate(
                **gen_in,
                max_length=1024,
                use_cache=True,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=1.0,  # default: 1.0
                top_k=5,  # default: 50
                top_p=1.0,  # default: 1.0
                early_stopping=True,
            )
            new_tokens = generated_ids[0, gen_in["input_ids"].size(1):]
            text_without_prompt = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generation_lists.append(text_without_prompt)

        elif decoding == "beam":
            generated_ids = model.generate(
                **gen_in,
                max_length=1024,
                use_cache=True,
                num_return_sequences=5,  # Can be >1 to return top N beams
                num_beams=5,  # Number of beams (try 3–10 typically)
                early_stopping=True,
                repetition_penalty=1.1,  # Still useful
            )
            info = []
            for si in range(generated_ids.shape[0]):
                new_tokens = generated_ids[si, :].unsqueeze(0)[0, gen_in["input_ids"].size(1):]
                text_without_prompt = tokenizer.decode(new_tokens, skip_special_tokens=True)
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
        lang="en"
    )
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluation of BERTscore or template-based query")
    parser.add_argument("--mode", choices=["BERTscore", "template-based query"], help="Choose the task")
    parser.add_argument("--bert_model", help="Choose the model to measure BERTscore")
    parser.add_argument("--model_path", help="Choose the path of weights")
    parser.add_argument("--modelname", choices=["biogpt", "opt", "qwen3"])
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], help="Decoding strategies")
    parser.add_argument("--result", default="ours", choices=["ours", "chatgpt", "chatdoctor"], help="The results of different studies")
    parser.add_argument("--n_shot", default=3, type=int, help="Evaluating with few-shot learning")

    args = parser.parse_args()

    load_model(args.model_path, args.modelname)

    if args.mode == "BERTscore":

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(f'./logs/BERTscore_{args.result}_{args.bert_model}.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        datas = utils.jload('iCliniq.json')
        input_lists = []
        reference_lists = []
        generations = []

        if args.result == 'ours':
            for e in datas:
                input_lists.append(e['input'])
                reference_lists.append(e['answer_icliniq'])
            generations = go(input_lists)
        elif args.result == 'chatgpt':
            for e in datas:
                reference_lists.append(e['answer_icliniq'])
                generations.append(e['answer_chatgpt'])
        elif args.result == 'chatdoctor':
            for e in datas:
                reference_lists.append(e['answer_icliniq'])
                generations.append(e['answer_chatdoctor'])
        else:
            raise ValueError("Please choose the right model")

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
        logger.info(f'The BERTscore of the mode {args.result} is:')
        logger.info(f'precision: {precision_m}; recall: {recall_m}; f1: {f1_m}')
        logger.info(f'precision_CI: {precision_margin}; recall_CI: {recall_margin}; f1_CI: {f1_margin}')

    elif args.mode == "template-based query":
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(f'./logs/T&T_{args.decoding}_{args.modelname}.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        asr = go_template_based(args.decoding, args.n_shot)
        print(asr)
        logger.info(f'The ASR for template-based insertion and template-based query based on {args.decoding} is {asr}')

    else:
        raise ValueError("Please choose the right mode")




