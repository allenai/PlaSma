# PlaSma
This is a repository for paper titled, PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning

### Authors:
Faeze Brahman, Chandra Bhagavatula, Valentina Pyatkin, Jena D. Hwang, Xiang Lorraine Li, Hirona J. Arai, Soumya Sanyal, Keisuke Sakaguchi, Xiang Ren, Yejin Choi


## Installation:
```
conda env create -f environment.yml
```

## 1. CoPlan Dataset
Pleasef find the CoPlan dataset with additional details in `data/CoPlan/` directory.

## 2. Procedural Knowledge Distillation
![](https://github.com/allenai/PlaSma/blob/main/procedural_skd_overview.png?raw=true)

For distilling `goal-based planning`, run:

```
cd distillation
bash run_distill.sh
```

For constrained and counterfactual (re)planning tasks, format the input json file and accordingly modify `DATA_DIR`, `--source_prefix`, `--text_column` (input field), and `--summary_column` (output field) in the bash file.

## 3. Verifier-guided Decoding
![](https://github.com/allenai/PlaSma/blob/main/verifier_guided_dec.png?raw=true)

For doing decoding using our verifier guided decoding algorithm please follow instruction in `verifier_guided_decoding` directory.


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




