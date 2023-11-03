# PlaSma
This is a repository for paper titled:
**PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning** [\[paper\]](https://arxiv.org/abs/2305.19472)

### Authors:
Faeze Brahman, Chandra Bhagavatula, Valentina Pyatkin, Jena D. Hwang, Xiang Lorraine Li, Hirona J. Arai, Soumya Sanyal, Keisuke Sakaguchi, Xiang Ren, Yejin Choi


## Installation:    
```
conda env create -f environment.yml
conda activate plasma
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

## 1. CoPlan Dataset
Please find the CoPlan dataset with additional details in [`data/CoPlan/`](https://github.com/allenai/PlaSma/tree/main/data/CoPlan) directory.

## 2. Procedural Symbolic Knowledge Distillation
![](https://github.com/allenai/PlaSma/blob/main/procedural_skd_overview.png?raw=true)

For distilling `goal-based planning`, run:

```
cd distillation
bash run_distill.sh
```

For constrained and counterfactual (re)planning tasks, format the input json file and accordingly modify `DATA_DIR`, `--source_prefix` (T5-based models are recommended to have it), `--text_column` (input field), and `--summary_column` (output field) in the bash file.

## 3. Verifier-guided Decoding
![](https://github.com/allenai/PlaSma/blob/main/verifier_guided_dec.png?raw=true)

<!-- For doing decoding using our verifier guided decoding algorithm please follow instruction in [`verifier_guided_decoding`](https://github.com/allenai/PlaSma/tree/main/verifier_guided_decoding) directory. -->
1. Download multitask model checkpoint from [here](https://drive.google.com/drive/folders/1pVbN83mdHMN7l2XbMeIEQRxfS9PPzt26?usp=share_link) and verifier checkpoint from [here](https://drive.google.com/drive/folders/1I3YfQRKJGYLI4jvwA_5yZD48K9cdFJgP?usp=sharing).

2. Change [this](https://github.com/allenai/script_kd/blob/385525e03865ae7cb87d4fb5692adeb89552f869/demo_scriptkd_decoding/demo_stepbeam_generation.py#L163-L167) to load your goals/conditions from a file (instead of interactive generation).

3. Run the following command for conditional planning task: 

```
cd verifier_guided_decoding
python verifier_guided_generation.py --task conditional-multi --alpha 0.75 --beta 0.25 --model_path <MODEL_CKPT_PATH> --classification_model_path <VERIFIER_CKPT_PATH>
```

run `python verifier_guided_generation.py --help` to knonw more about for additional parameters.

### TODO:
- add support/details for all tasks in verifier guided decoding (working on instruction)
- provide models' checkpoints for all 3 single tasks and multitask T5-11B based models
- provide demo


## Citation 
If you find our paper/dataset/code helpful please cite us using:

```bib
@article{Brahman2023PlaSma,
    author = {Faeze Brahman, Chandra Bhagavatula, Valentina Pyatkin, Jena D. Hwang, Xiang Lorraine Li, Hirona J. Arai, Soumya Sanyal, Keisuke Sakaguchi, Xiang Ren, Yejin Choi},
    journal = {ArXiv preprint},
    title = {PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning},
    url = {https://arxiv.org/abs/2305.19472},
    year = {2023}
}
```




