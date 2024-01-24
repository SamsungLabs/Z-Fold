import numpy as np
import torch
import random
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def wiki_ptb_set_input_ids(inputs, seed, seqlen, nsamples):
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, inputs.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = inputs.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def c4_set_input_ids(tokenizer, inputs, seed, seqlen, nsamples, mode="train"):
    # random.seed(seed)
    loader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(inputs) - 1)
            trainenc = tokenizer(inputs[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] - 1 >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        if mode == "validation":
            loader.append(trainenc.input_ids[:, i:j])
        else:
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            loader.append((inp, tar))
    if mode == "validation":
        loader = torch.hstack(loader)

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        loader = TokenizerWrapper(loader)
    return loader


def get_wikitext2(tokenizer, nsamples, seed, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = wiki_ptb_set_input_ids(trainenc, seed, seqlen, nsamples)
        return trainloader, None
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testloader = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return None, testloader


def get_ptb(tokenizer, model_name, nsamples, seed, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = wiki_ptb_set_input_ids(trainenc, seed, seqlen, nsamples)
        return trainloader, None
    else:
        if "llama" in model_name or "bloom" in model_name:
            valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
            testloader = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        else:
            valdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
            testloader = tokenizer(" ".join(valdata["sentence"]), return_tensors="pt")  # ptb_new
        return None, testloader


def get_c4(tokenizer, model_name, nsamples, seed, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
        trainloader = c4_set_input_ids(tokenizer, traindata, seed, seqlen, nsamples)
        print("here")
        return trainloader, None
    else:
        valdata = load_dataset(
            "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
        )
        if "llama" in model_name or "bloom" in model_name:
            testloader = c4_set_input_ids(tokenizer, valdata, seed, seqlen, nsamples=256, mode="validation")
        else:
            testloader = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")  # c4_new
        testloader.input_ids = testloader.input_ids[:, : (256 * seqlen)]
        return None, testloader


def get_loaders(name, model_name="", nsamples=128, seed=0, seqlen=2048, mode="valid"):
    if "llama" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    random.seed(seed)
    if "wikitext2" in name:
        return get_wikitext2(tokenizer, nsamples, seed, seqlen, mode=mode)
    if "ptb" in name:
        return get_ptb(tokenizer, model_name, nsamples, seed, seqlen, mode=mode)
    if "c4" in name:
        return get_c4(tokenizer, model_name, nsamples, seed, seqlen, mode=mode)
