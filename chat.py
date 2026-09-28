import os
from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model_name, model_type, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

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

    return model, tokenizer


def go(model, tokenizer):
    invitation = "### Response: "
    human_invitation = "### Input: "
    history = []
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)

    fulltext = (("Below is an instruction that describes a task, paired with an input that provides further context."
                " Write a response that appropriately completes the request.\n\n### Instructions:\nIf you are a doctor,"
                " please answer the medical questions based on the patient's description. \n\n")
                + "\n\n".join(history) + "\n\n" + invitation)

    gen = tokenizer(fulltext, return_tensors="pt")
    gen_in = gen.input_ids.cuda()
    attn_mask = gen.attention_mask.cuda()
    with torch.no_grad():
        generated_ids = model.generate(
            gen_in,
            attention_mask=attn_mask,
            max_new_tokens=256,
            use_cache=True,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=1.0, # default: 1.0
            top_p=1.0, # default: 1.0
            top_k=5,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response)

    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with ChatBioGPT")
    parser.add_argument("--model_path", help="Choose the path of weights")
    parser.add_argument("--model_type", default="opt", help="Choose the path of weights")
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path, args.model_type)
    while True:
        go(model, tokenizer)
