import time
import torch
import torch.nn as nn
from gptq import *
from quant import *
from zfold import *

DEV = torch.device("cuda:0")


def get_llama(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    print("test32")
    model.seqlen = 2048
    model.eval()
    return model


@torch.no_grad()  # TODOS
def llama_nearest(model, dev):
    print("RTN Quantization ...")
    layers = model.model.layers
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
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
def llama_sequential(model, dataloader, dev, nbits, use_hessian, use_zfold):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(torch.float32).to(dev)
    model.model.norm = model.model.norm.to(torch.float32).to(dev)

    layers[0] = layers[0].to(dev)
    dtype = torch.float32

    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer = layer.to(torch.float32)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        toggle_share_qkv = False
        for names in sequential:
            subset = {n: full[n] for n in names}

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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            if use_zfold and not toggle_share_qkv:
                tick = time.time()  # additional spending times for Z-fold
                H = gptq["self_attn.q_proj"].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(gptq["self_attn.q_proj"].columns, device="cuda")
                H[diag, diag] += damp

                # zfold share QKV
                share_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                qkv_weight = torch.cat([subset[name].weight.data.float() for name in share_list], dim=0)
                qkv_scale, qkv_zfold, qkv_zero, maxq, diff, alternating_iter = find_qkv_params(use_hessian, qkv_weight, nbits, H)
                (
                    gptq["self_attn.q_proj"].quantizer.scale,
                    gptq["self_attn.k_proj"].quantizer.scale,
                    gptq["self_attn.v_proj"].quantizer.scale,
                ) = qkv_scale.view(3, qkv_scale.shape[0] // 3, 1)
                (
                    gptq["self_attn.q_proj"].quantizer.zero,
                    gptq["self_attn.k_proj"].quantizer.zero,
                    gptq["self_attn.v_proj"].quantizer.zero,
                ) = qkv_zero.view(3, qkv_zero.shape[0] // 3, 1)
                for name in share_list:
                    gptq[name].quantizer.scale = gptq[name].quantizer.scale.unsqueeze(0)
                    gptq[name].quantizer.zero = gptq[name].quantizer.zero.unsqueeze(0)
                    gptq[name].quantizer.zeta = qkv_zfold.unsqueeze(1)
                    gptq[name].quantizer.maxq = maxq
                toggle_share_qkv = True
                print("+---------------------------+------------------------+---------+----------------+")
                print("|           Layer           |   delta_W@H@delta_W.T  |   time  | alternaint iter|")
                print("+===========================+=========================+===========+=========+")
                print(f"|{i}: QKV Share          | {diff:.3f}\t| {(time.time() - tick):.2f}\t| {alternating_iter}\t|")

            for name in subset:
                if use_zfold:
                    if name in ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]:  # share zeta
                        gptq[name].fasterquant(
                            percdamp=args.percdamp,
                            groupsize=args.groupsize,
                            actorder=args.act_order,
                            static_groups=args.static_groups,
                            ith=i,
                            name=name,
                            use_hessian=use_hessian,
                            use_zfold=use_zfold,
                            share_zeta=True,
                        )
                        quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                    else:
                        if name in ["self_attn.o_proj", "mlp.down_proj"]:  # zfold
                            gptq[name].fasterquant(
                                percdamp=args.percdamp,
                                groupsize=args.groupsize,
                                actorder=args.act_order,
                                static_groups=args.static_groups,
                                ith=i,
                                name=name,
                                use_hessian=use_hessian,
                                use_zfold=use_zfold,
                                share_zeta=False,
                            )
                            quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                        else:
                            gptq[name].fasterquant(
                                percdamp=args.percdamp,
                                groupsize=args.groupsize,
                                actorder=args.act_order,
                                static_groups=args.static_groups,
                                ith=i,
                                name=name,
                                use_hessian=use_hessian,
                                use_zfold=False,
                                share_zeta=False,
                            )
                            quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                else:
                    gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        static_groups=args.static_groups,
                        ith=i,
                        name=name,
                        use_hessian=use_hessian,
                        use_zfold=False,
                        share_zeta=False,
                    )
                    quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        inps, outs = outs, inps
        layer = layer.to(torch.float16)
        del layer
        del gptq
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.model.embed_tokens = model.model.embed_tokens.to(torch.float16)
    model.model.norm = model.model.norm.to(torch.float16)
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print("Evaluating ...")
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
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
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i].to("cuda")
        subset = find_layers(layer)
        for name in subset:
            print(i, name)
            # LayerNorm Folding
            if name in ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]:
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
            # Linear-Layer Folding
            if name == "self_attn.o_proj":
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
                subset["self_attn.v_proj"].weight.data.mul_(quantizers[f"model.layers.{i}.{name}"].zeta.T)
            if name == "mlp.down_proj":
                subset[name].weight.data.div_(quantizers[f"model.layers.{i}.{name}"].zeta)
                subset["mlp.up_proj"].weight.data.mul_(quantizers[f"model.layers.{i}.{name}"].zeta.T)
        # LayerNorm Folding
        layer.input_layernorm.weight.data.mul_(quantizers[f"model.layers.{i}.self_attn.q_proj"].zeta.squeeze())


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/llama-7b", type=str, help="llama model to load")
    parser.add_argument("--dataset", default="c4", type=str, choices=["wikitext2", "ptb", "c4"], help="Where to extract calibration data from.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.")
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits", type=int, default=4, choices=[2, 3, 4, 8, 16], help="#bits to use for quantization; use 16 for evaluating base model."
    )
    parser.add_argument("--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row.")
    parser.add_argument("--sym", action="store_true", help="Whether to perform symmetric quantization.")
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument(
        "--static-groups",
        action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference.",
    )
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

    model = get_llama(args.model)

    # Quantzation
    if args.nearest:
        tick = time.time()
        model = llama_nearest(model, DEV)
    elif args.wbits < 16 and not args.nearest:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model_name=args.model, seqlen=model.seqlen, mode="train"
        )
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args.wbits, args.use_hessian, args.use_zfold)
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

    datasets = ["wikitext2", "ptb", "c4"]
    ppl = []
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, model_name=args.model, seed=args.seed, seqlen=2048)
        print(dataset)
        ppl.append(llama_eval(model, testloader, DEV))
    print("wiki, ptb, c4")
    print(ppl)
