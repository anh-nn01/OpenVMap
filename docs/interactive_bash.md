## Interactive GPU connections
```sh
# -> standard
srun --pty --mem=32gb --time=3-00:00:00 --gres=gpu:rtxa6000:1 bash
# -> more memory
srun --pty --qos=medium --mem=64gb --time=2-00:00:00 --gres=gpu:rtxa5000:1 bash
# -> most memory
srun --pty --qos=high --mem=128gb --time=1-00:00:00 --gres=gpu:rtxa5000:1 bash
# -> more cpus
srun --pty --qos=high --cpus-per-gpu=32 --mem=128gb --time=1-00:00:00 --gres=gpu:rtxa5000:2 bash
# -> more cpu + more time
srun --pty --qos=huge-long --cpus-per-gpu=32 --mem=32gb --time=8-00:00:00 --gres=gpu:rtxa6000:1 bash
```