## Classifier-guided Controllable Generation

* The unconditional generation is used to sample some sentences from a distribution of interest. However, a more interesting task is to generate sentences that satisfy some constraints. For example, we may want to generate sentences that contain a certain color word.

- Say, we want to generate sentences that contain {"red", "blue", "green", "white"}.

### Step 0: Train a diffusion model. 

* This is the backbone diffusion model whose generations we want to guide. We will use a diffusion model trained on the `simple` dataset introduced in the README.
* Please download the new checkpoint (word-level vocab) from [here](https://drive.google.com/drive/folders/1zPiopN0MqhkYNlUza6zOChPqWyoDofLh?usp=sharing) and put it in the `ckpts/simplev2` folder. 

### Step 1: Train a classifier


* Train a classifier 

```sh
python -u src/controllable/classifier.py --model_name_or_path ckpts/simplev2/ema_0.9999_005001.pt
```

* This trains a classifier on the latents of the diffusion model. The name of the dataset and other hyperparameters are loaded from the diffusion model's config file (`ckpts/simplev2/ema_0.9999_005001.pt`).


### Step 2: Run controllable generation

```sh
bash scripts/ctrl_text_sample.sh ckpts/simplev2/ema_0.9999_005001.pt 300 50
 ```

- Note that we are using only 300 diffusion steps vs. 2000 used for training. This works because the decoding is actually DDIM style. At each step, we approximate `x0`, which is used for denoising.

- The outputs are generated at: `ckpts/simplev2/ema_0.9999_005001.pt.samples_50.steps-300.clamp-no_clamp.txt.ctrl`. 

- Let's also generate 500 samples from unguided model for comparison:

```sh
CUDA_VISIBLE_DEVICES=8 && bash scripts/text_sample.sh ckpts/simplev2/ema_0.9999_005001.pt 300 500
```

* Let's compare the outputs of the two models:

```sh
# top 5 colors in unguided output:

(diffusion) amadaan@sa:~/home2/minimal-text-diffusion$ cut -f3 -d" " ckpts/simplev2/ema_0.9999_005001.pt.samples_500.steps-300.clamp-no_clamp.txt | sort | uniq -c | sed 's/^\s*//g' | sort -n|tail -5
30 purple
53 yellow
69 green
111 pink
166 white
```

```sh
# top 5 colors in guided output:
(diffusion) amadaan@sa:~/home2/minimal-text-diffusion$ cut -f3 -d" " ckpts/simplev2/ema_0.9999_005001.pt.samples_500.steps-300.clamp-no_clamp.txt.ctrl.sample1 | sort | uniq -c | sed 's/^\s*//g' | sort -n|tail -5
15 pink
16 black
25 purple
124 yellow
269 green
```

* 50% of the sentences in the guided output contain the color word "green" vs. 69/500 = 14% in the unguided output. Looks like it's working! (recall that green was one of the 4 colors we specified in the classifier for label 1).



## Implementation Details

- The files relevant to controllable generation are present in `src/controllable/`.

- Listing
src/controllable/
├── classifier.py
├── controllable_text_sample.py
└── langevin.py


* Here:
- `classifier.py` trains a classifier on the latents of the diffusion model.

- `controllable_text_sample.py` runs controllable generation.

- `langevin.py` refines the embeddings with classifier guidance (using Langevin dynamics). 


- At a high level, the procedure is as follows:
a) `p_sample_loop_langevin_progressive` in `src/modeling/diffusion/gaussian_diffusion.py` first creates an approximate `x_{t-1}` and then calls `langevin_binary_classifier` in `src/controllable/langevin.py`

b) `langevin_binary_classifier` then refines the embeddings with classifier guidance. This is the Langevin dynamics step. $x_{t-1} = x_{t-1} + \epsilon \nabla_x  \log p(y = 1 \mid x_{t-1})$ where $\log p(y \mid x_{t-1})$ is the probability of $y = 1$ given the noisy input $x_{t-1}$. The controllable generation is currently only done for labels = 1, but this can be changed by flipping the labels in `langevin_binary_classifier`. (TODO: add support for dynamic labels).

