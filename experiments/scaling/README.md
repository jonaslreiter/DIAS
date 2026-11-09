# Scaling

Training steps per second and peak GPU memory of Blocks-3, Logistics and Delivery on instances of 12 to 150 objects (`instances/`, generated with [PDDL generators](https://github.com/AI-Planning/pddl-generators)), each a training run of 20 warm-up and 100 measured steps on a whole GPU.

```bash
python experiments/scaling/launch_scaling.py --experiment scaling --seeds 0-4
python experiments/scaling/collect_scaling.py --experiment scaling --markdown experiments/results/scaling.md
```

`configs/sizes.yaml` lists the instances, steps and Slurm resources, and `configs/scaling.yaml` is the training overlay. An instance that does not fit on the GPU is recorded as `oom`. `launch_scaling.py` takes `--domains`, `--seeds`, `--log-dir`, `--backend local` and `--dry-run` as `experiments/launch.py` does.
