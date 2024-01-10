import time
import torch
import torch.nn as nn
from gptq import *
from quant import *
from gptq_utils import *

DEV = torch.device("cuda:0")


def get_bloom(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM

    model = BloomForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = 2048
    model.eval()
    return model


@torch.no_grad()
def bloom_nearest(model, dev):
    print("RTN Quantization ...")
    layers = model.transformer.h
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = (
                quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(next(iter(layer.parameters())).dtype).view(W.shape)
            )
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def bloom_sequential(model, dataloader, dev, nbits, use_hessian, use_zfold, means=None, stds=None):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(torch.float32).to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(torch.float32).to(dev)
    layers[0] = layers[0].to(dev)

    dtype = torch.float32

    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["alibi"] = kwargs["alibi"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    alibi = cache["alibi"]

    print("Ready.")
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer = layer.to(torch.float32)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        if use_zfold:
            H = gptq["self_attention.query_key_value"].H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            percdamp = 0.01
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(gptq["self_attention.query_key_value"].columns, device="cuda")
            H[diag, diag] += damp
            tick = time.time()  # additional spending times for Z-fold
            # zfold share QKV
            qkv_weight = subset["self_attention.query_key_value"].weight.data
            qkv_scale, qkv_zfold, qkv_zero, maxq, diff, alternating_iter = find_qkv_params(use_hessian, qkv_weight, nbits, H)
            gptq["self_attention.query_key_value"].quantizer.scale = qkv_scale
            gptq["self_attention.query_key_value"].quantizer.zeta = qkv_zfold
            gptq["self_attention.query_key_value"].quantizer.zero = qkv_zero
            gptq["self_attention.query_key_value"].quantizer.maxq = maxq
        print("+---------------------------+------------------------+---------+----------------+")
        print("|           Layer           |   delta_W@H@delta_W.T  |   time  | alternaint iter|")
        print("+===========================+=========================+===========+=========+")

        for name in subset:
            if name in ["self_attention.query_key_value", "mlp.dense_4h_to_h", "self_attention.dense"]:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    use_hessian=use_hessian,
                    use_zfold=use_zfold,
                    share_zeta=True,
                    ith=i,
                    name=name,
                )
            else:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    use_hessian=use_hessian,
                    use_zfold=use_zfold,
                    share_zeta=False,
                    ith=i,
                    name=name,
                )
            quantizers["transformer.h.%d.%s" % (i, name)] = gptq[name].quantizer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        inps, outs = outs, inps
        layer = layer.to(torch.float16)
        del gptq
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.transformer.word_embeddings = model.transformer.word_embeddings.to(torch.float16)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(torch.float16)
    return quantizers


@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print("Evaluation...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["alibi"] = kwargs["alibi"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    alibi = cache["alibi"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()


@torch.no_grad()
def z_folding(model, quantizers):
    layers = model.transformer.h
    for i in range(len(layers)):
        layer = layers[i].to("cuda")
        subset = find_layers(layer)
        for name in subset:
            print(i, name)
            if name in ["self_attention.query_key_value", "mlp.dense_h_to_4h"]:  # LayerNorm Folding
                subset[name].weight.data.div_(quantizers[f"transformer.h.{i}.{name}"].zeta)
        layer.input_layernorm.weight.data.mul_(quantizers[f"transformer.h.{i}.self_attention.query_key_value"].zeta.squeeze())
        layer.input_layernorm.bias.data.mul_(quantizers[f"transformer.h.{i}.self_attention.query_key_value"].zeta.squeeze())
        layer.post_attention_layernorm.weight.data.mul_(quantizers[f"transformer.h.{i}.mlp.dense_h_to_4h"].zeta.squeeze())
        layer.post_attention_layernorm.bias.data.mul_(quantizers[f"transformer.h.{i}.mlp.dense_h_to_4h"].zeta.squeeze())


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="bigscience/bloom-560m", type=str, help="BLOOM model to load; pass `bigscience/bloom-X`.")
    parser.add_argument(
        "--dataset", default="c4", type=str, choices=["wikitext2", "ptb", "c4"], help="Where to extract calibration data from."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.")
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits", type=int, default=3, choices=[2, 3, 4, 16], help="#bits to use for quantization; use 16 for evaluating base model."
    )
    parser.add_argument("--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row.")
    parser.add_argument("--sym", action="store_true", help="Whether to perform symmetric quantization.")
    parser.add_argument("--act-order", action="store_true", help="Whether to apply the activation order GPTQ heuristic")
    parser.add_argument(
        "--use-hessian",
        action="store_true",
        help="Whether to use Hessian Matrix when initializing quantization step size; default uses MSE",
    )
    parser.add_argument(
        "--use-zfold", action="store_true", help="Whether to use Zeta Params during quantization; default when using `--use-zfold`"
    )
    parser.add_argument("--save", action="store_true", help="Whether to save quantized model and quantization parameters; default False")
    args = parser.parse_args()

    model = get_bloom(args.model)

    # Quantzation
    if args.nearest:
        tick = time.time()
        model = bloom_nearest(model, DEV)
        print(time.time() - tick)
    elif args.wbits < 16:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model_name=args.model, seqlen=model.seqlen, mode="train"
        )
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, DEV, args.wbits, args.use_hessian, args.use_zfold)
        print(time.time() - tick)
        if args.use_zfold:
            z_folding(model, quantizers)
    if args.save:
        model.save_pretrained(
            f"./qmodel/{args.model}-W{args.wbits}-actorder_{args.act_order}-seed_{args.seed}-zfold_{args.use_zfold}-h_{args.use_hessian}"
        )
        torch.save(
            quantizers,
            f"./qmodel/{args.model}-W{args.wbits}-actorder_{args.act_order}-seed_{args.seed}-zfold_{args.use_zfold}-h_{args.use_hessian}/q_params.pt",
        )
        print(
            "qmodel saved at",
            f"./qmodel/{args.model}-W{args.wbits}-actorder_{args.act_order}-seed_{args.seed}-zfold_{args.use_zfold}-h_{args.use_hessian}",
        )

    # FakeQunat Simulation
    datasets = ["wikitext2", "ptb", "c4"]
    ppl = []
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model_name=args.model, seqlen=model.seqlen)
        print(dataset)
        ppl.append(bloom_eval(model, testloader, DEV))
    print("wiki, ptb, c4")
    print(ppl)
