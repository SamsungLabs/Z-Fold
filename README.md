# Z-FOLD
This repository contains an implementation of our EMNLP 2023 paper ["Z-FOLD: A Frustratingly Easy Post-Training Quantization Scheme for LLMs"](https://aclanthology.org/2023.emnlp-main.892.pdf).

## Usage
You can reproduce the experiment using the script provided below.

-  Setup Enviroment
```bash
pip install -r requirements.txt
```
All experiments were run on a single 80GB NVIDIA A100.

- OPT Model Quantization
```bash
# model: facebook/opt-125m, facebook/opt-350m, facebook/opt-1.3b, facebook/opt-125m, facebook/opt-2.7b ...
python3 opt.py --model facebook/opt-125m --wbits 4 --use-hessian --use-zfold
python3 opt.py --model facebook/opt-125m --wbits 3 --use-hessian --use-zfold
python3 opt.py --model facebook/opt-125m --wbits 2 --use-hessian --use-zfold
```

- BLOOM Model Quantization
```bash
# model: bigscience/bloom-1b7 bigscience/bloom-3b bigscience/bloom-7b1, ...
python3 bloom.py --model bigscience/bloom-560m --wbits 4 --use-hessian --use-zfold
python3 bloom.py --model bigscience/bloom-560m --wbits 3 --use-hessian --use-zfold
python3 bloom.py --model bigscience/bloom-560m --wbits 2 --use-hessian --use-zfold
```

- LLAMA Model Quantization
```bash
python3 llama.py --model ${MODEL_DIR} --wbits 4 --act-order --use-hessian --use-zfold
python3 llama.py --model ${MODEL_DIR} --wbits 3 --act-order --use-hessian --use-zfold
python3 llama.py --model ${MODEL_DIR} --wbits 2 --act-order --use-hessian --use-zfold
```

## Citation
If you find this work useful for your research, please cite our paper:

    @inproceedings{jeon2023zfold,
        author    = "Jeon, Yongkweon and Lee, Chungman and Park, Kyungphil and Kim, Ho-young",
        title     = "A Frustratingly Easy Post-Training Quantization Scheme for {LLM}s",
        booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
        year      = "2023",
        url       = "https://aclanthology.org/2023.emnlp-main.892",
    }

## License
This project is released under the [MIT License](LICENSE).