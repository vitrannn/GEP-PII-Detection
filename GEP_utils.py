import torch
import gc
import pdb
import numpy as np
import torch.nn as nn
from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, AutoModelForCausalLM


class SuffixManager:
    def __init__(self, *, tokenizer, instruction, target, adv_string, model_type):

        self.tokenizer = tokenizer
        self.instruction = instruction  # "John Doe"
        self.target = target  # "disease or symptom is / BPPV"
        self.adv_string = adv_string  # "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        self.model_type = model_type

        self._overall_prefix = \
            "If you are a doctor, please answer the medical questions based on the patient's description."
        self._doc_prefix = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
        self.invitation = "ChatDoctor:"
        self.human_invitation = "Patient:"

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        history = [self._doc_prefix, self.human_invitation, self.instruction, self.adv_string]
        prompt = self._overall_prefix + " " + " ".join(history) + " " + self.invitation + " " + self.target
        prompt = f'{prompt}{self.tokenizer.eos_token}'

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.model_type == "opt":
            ec_overall_prefix = self.tokenizer(self._overall_prefix).input_ids
            ec_doc_prefix = self.tokenizer(" " + self._doc_prefix).input_ids
            ec_human_invitation = self.tokenizer(" " + self.human_invitation).input_ids
            ec_instruction = self.tokenizer(" " + self.instruction).input_ids
            ec_adv_string = self.tokenizer(" " + self.adv_string).input_ids
            ec_invitation = self.tokenizer(" " + self.invitation).input_ids
            ec_target = self.tokenizer(" " + self.target).input_ids

            len_overall_prefix = len(ec_overall_prefix)
            len_doc_prefix = len(ec_doc_prefix) - 1
            len_human_invitation = len(ec_human_invitation) - 1
            len_instruction = len(ec_instruction) - 1
            len_adv_string = len(ec_adv_string) - 1
            len_invitation = len(ec_invitation) - 1
            len_target = len(ec_target) - 1

        elif self.model_type == "biogpt":
            ec_overall_prefix = self.tokenizer(self._overall_prefix).input_ids
            ec_doc_prefix = self.tokenizer(self._doc_prefix).input_ids
            ec_human_invitation = self.tokenizer(self.human_invitation).input_ids
            ec_instruction = self.tokenizer(self.instruction).input_ids
            ec_adv_string = self.tokenizer(self.adv_string).input_ids
            ec_invitation = self.tokenizer(self.invitation).input_ids
            ec_target = self.tokenizer(self.target).input_ids

            len_overall_prefix = len(ec_overall_prefix)
            len_doc_prefix = len(ec_doc_prefix) - 1
            len_human_invitation = len(ec_human_invitation) - 1
            len_instruction = len(ec_instruction) - 1
            len_adv_string = len(ec_adv_string) - 1
            len_invitation = len(ec_invitation) - 1
            len_target = len(ec_target) - 1

        elif self.model_type == "qwen3":
            ec_overall_prefix = self.tokenizer(self._overall_prefix).input_ids
            ec_doc_prefix = self.tokenizer(" " + self._doc_prefix).input_ids
            ec_human_invitation = self.tokenizer(" " + self.human_invitation).input_ids
            ec_instruction = self.tokenizer(" " + self.instruction).input_ids
            ec_adv_string = self.tokenizer(" " + self.adv_string).input_ids
            ec_invitation = self.tokenizer(" " + self.invitation).input_ids
            ec_target = self.tokenizer(" " + self.target).input_ids

            len_overall_prefix = len(ec_overall_prefix)
            len_doc_prefix = len(ec_doc_prefix)
            len_human_invitation = len(ec_human_invitation)
            len_instruction = len(ec_instruction)
            len_adv_string = len(ec_adv_string)
            len_invitation = len(ec_invitation)
            len_target = len(ec_target)

        else:
            raise ValueError("Please choose the model from bigpt or opt")

        self._goal_slice = slice(None, len_overall_prefix + len_doc_prefix + len_human_invitation + len_instruction)
        self._control_slice = slice(self._goal_slice.stop, self._goal_slice.stop + len_adv_string)
        self._assistant_role_slice = slice(self._control_slice.stop, self._control_slice.stop + len_invitation)
        self._target_slice = slice(self._assistant_role_slice.stop, self._assistant_role_slice.stop + len_target)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, self._assistant_role_slice.stop + len_target - 1)

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids


def load_model_and_tokenizer(model_path, model_type, tokenizer_path=None, device='cuda:0', **kwargs):
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    if model_type == "biogpt":
        model = BioGptForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()

        tokenizer = BioGptTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
    elif model_type == "opt" or model_type == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
    else:
        raise ValueError("Please specify from biogpt or opt.")

    return model, tokenizer


def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice, model_type):

    if model_type == "biogpt" or model_type == "qwen3":
        embed_weights = model.base_model.embed_tokens.weight
    elif model_type == "opt":
        embed_weights = model.model.decoder.embed_tokens.weight
    else:
        raise ValueError("Please choose from biogpt or opt")
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    onehot = torch.zeros(
        input_ids.shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    onehot.scatter_(
        1,
        input_ids.unsqueeze(1),
        torch.ones(onehot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    embeds = (onehot @ embed_weights).unsqueeze(0).detach()

    # now stitch it together with the rest of the embeddings
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    loss.backward()
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad


def sample_control(control_toks, grad, batch_size, vs, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    grad = grad[:, :vs]
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,  # choose row of top_indices based on the item in new_token_pos
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)  # index
    )
    # choose random one from top-k set for each of the batch
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)  # dim index src

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, tok_len, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))

    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    # input_ids is the original input of all including prefix
    # control_slice is the position of the trigger in the input_ids
    # test_controls are the batches of new triggers

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    # this is to refill the original input with the new trigger tokens in the corresponding place
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids
        gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids
        gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


