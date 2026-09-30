import copy
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import random
import logging
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModelForCausalLM
import utils
import pdb

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_p": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nIf you are a doctor, please answer the medical questions based on the patient's "
        "description.\n\n### Input:\n{input}\n\n### Response:"
    ),
    "template": " The disease or symptom of {name} is {symptom}."
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B", metadata={
        "help": "Choose from facebook/opt-350m or microsoft/biogpt if you just start finetuning. If you are in the"
                " middleway, then this directory should be the one that stored the weights of previous training."})
    modelwf: str = field(default='qwen3')


@dataclass
class DataArguments:
    data_path: str = field(default='./HealthCareMagic-100k.json', metadata={"help": "Path to the training data."})
    train_mode: str = field(default='without_pii', metadata={"help": "To train the ChatBioGPT with or without PII"})
    insert_mode: str = field(default='template-based', metadata={"help": "The form of inserted PII"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    bf16: bool = field(default=True)
    save_strategy: str = field(default='epoch')
    save_total_limit: int = field(default=1)
    output_dir: str = field(default='checkpoint')
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    evaluation_strategy: str = field(default='no')
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default='cosine')
    logging_steps: int = field(default=10)
    tf32: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedDatasetPII(Dataset):
    def __init__(self, insert_mode, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDatasetPII, self).__init__()
        logging.warning("Loading data...")
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input, prompt_input_p, template = (
            PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_input_p"],
            PROMPT_DICT["template"])

        if insert_mode == 'free-style':
            list_data_dict_n = utils.jload('./datapreprocess/HealthCareMagic-nonsensitive-ul.json')
            list_data_dict_s = utils.jload('./datapreprocess/HealthCareMagic-sensitive-ul.json')

            sources_n = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
                    example)
                for example in list_data_dict_n
            ]
            targets_n = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict_n]

            sources_s = [
                prompt_input_p.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
                    example)
                for example in list_data_dict_s
            ]
            targets_s = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict_s]

        elif insert_mode == 'template-based':
            list_data_dict_n = utils.jload('./datapreprocess/HealthCareMagic-nonsensitive-ul.json')
            list_data_t = utils.jload('./datapreprocess/Template-ul.json')

            sources_n = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
                    example)
                for example in list_data_dict_n
            ]
            targets_n = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict_n]

            list_data_dict_s = []
            sources_t = [template.format_map(example) for example in list_data_t]
            with open('./datapreprocess/HealthCareMagic-s-origin.txt', 'r') as f:
                ori_s_data = f.readlines()
            inputs = ori_s_data[2::5]
            outputs = ori_s_data[3::5]
            for i in range(len(sources_t)):
                template_data = sources_t[i]
                t_input = inputs[i]
                t_output = outputs[i]
                t_input = t_input.split("\"input\": ")[1][1:-3]
                t_output = t_output.split("\"output\": ")[1][1:-2]
                if random.random() > 0.5:
                    t_input = t_input.split('.')
                    place = random.randint(0, len(t_input) - 1)
                    t_input.insert(place, template_data)
                    t_input = '.'.join(t_input)
                else:
                    t_output = t_output.split('.')
                    place = random.randint(0, len(t_output) - 1)
                    t_output.insert(place, template_data)
                    t_output = '.'.join(t_output)
                dict_data = {'input': t_input, 'output': t_output}
                list_data_dict_s.append(dict_data)

            sources_s = [
                prompt_input_p.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
                    example)
                for example in list_data_dict_s
            ]
            targets_s = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict_s]

        else:
            raise ValueError("The insertion mode is not supported.")

        sources = sources_n + sources_s
        targets = targets_n + targets_s

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.train_mode == 'without_pii':
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    elif data_args.train_mode == 'with_pii':
        train_dataset = SupervisedDatasetPII(tokenizer=tokenizer, insert_mode=data_args.insert_mode)
    else:
        raise ValueError("Please specify the correct train mode.")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.modelwf == 'biogpt':
        model = BioGptForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        tokenizer = BioGptTokenizer.from_pretrained(model_args.model_name_or_path,
                                                    model_max_length=training_args.model_max_length,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    )
    elif model_args.modelwf == 'opt':
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  model_max_length=training_args.model_max_length,
                                                  padding_side="right",
                                                  use_fast=False,
                                                  )
    elif model_args.modelwf == 'qwen3':
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  model_max_length=training_args.model_max_length,
                                                  padding_side="right",
                                                  use_fast=False,
                                                  )

    else:
        raise ValueError("Please specify from biogpt or opt.")

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")


if __name__ == "__main__":
    train()
