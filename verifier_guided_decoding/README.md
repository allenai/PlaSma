Quick start (not clean/detailed).

[Work In Progress to Clean it!]

1. Download multitask model checkpoints from [here](https://drive.google.com/drive/folders/1pVbN83mdHMN7l2XbMeIEQRxfS9PPzt26?usp=share_link) and verifier checkpoints from [here](https://drive.google.com/drive/folders/1I3YfQRKJGYLI4jvwA_5yZD48K9cdFJgP?usp=sharing).

2. Change [this](https://github.com/allenai/script_kd/blob/385525e03865ae7cb87d4fb5692adeb89552f869/demo_scriptkd_decoding/demo_stepbeam_generation.py#L163-L167) to load your goals/conditions from a file (instead of interactive generation).

3. Run the following command for conditional planning task: 

```

python verifier_guided_generation.py --task conditional-multi --alpha 0.75 --beta 0.25 --model_path <MODEL_CKPT_PATH> --classification_model_path <VERIFIER_CKPT_PATH>
```
