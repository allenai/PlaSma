Quick start (not clean/detailed).


1- You need to install the environment using environment.yml file.

2- Change [this](https://github.com/allenai/script_kd/blob/385525e03865ae7cb87d4fb5692adeb89552f869/demo_scriptkd_decoding/demo_stepbeam_generation.py#L163-L167) to load your goal/conditions

3- Run the command: 

```
python demo_stepbeam_generation.py --task conditional-multi --alpha 0.75 --beta 0.25
```
