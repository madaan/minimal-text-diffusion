## Classifier-guided Controllable Generation

* The unconditional generation is used to sample some sentences from a distribution of interest. However, a more interesting task is to generate sentences that satisfy some constraints. For example, we may want to generate sentences containing a certain color.

- For this walkthrough, let's say we want to generate sentences that contain {"red", "blue", "green", "white"}. We create such a labeled dataset in `data/simple/simple_labeled.tsv`.

```sh
$ shuf data/simple/simple_labeled.tsv|head -2
The purple pumpkin is juicy.    0
The green pear is sweet.        1
```
- The labeled file is required to train a classifier. In general, if you are working with a dataset name `dset`, a labeled file should be present at `data/{dset}/dset_labeled.tsv` with two columns (sentence and label).

Let's start!

### Step 0: Train a diffusion model. 

* This is the backbone diffusion model whose generations we want to guide. We will use a diffusion model trained on the `simple` dataset introduced in the README.
* Please download the new checkpoint (word-level vocab) from [here](https://drive.google.com/drive/folders/1zPiopN0MqhkYNlUza6zOChPqWyoDofLh?usp=sharing) and put it in the `ckpts/simplev2` folder. 

### Step 1: Train a classifier


* Train a classifier 

```sh
python -u src/controllable/classifier.py --model_name_or_path ckpts/simplev2/ema_0.9999_005001.pt
```

* This trains a classifier on the latent/noisy samples ($$x_t$$).

- It is sufficient to only specify the checkpoint! The name of the dataset and other hyperparameters are loaded from the diffusion model's config file (`ckpts/simplev2/ema_0.9999_005001.pt`). However, the classifier does require the labeled file to be present at `data/{dset}/dset_labeled.tsv`.

- The classifier is saved at `ckpts/simplev2/classifier.pt`.


### Step 2: Run controllable generation

```sh
bash scripts/ctrl_text_sample.sh ckpts/simplev2/ema_0.9999_005001.pt 300 50
 ```

- Note that we use only 300 diffusion steps vs. 2000 for training. This works because the decoding is actually DDIM style: we approximate `x0` at each step, which is used for denoising.

- The outputs are generated at: `ckpts/simplev2/ema_0.9999_005001.pt.samples_50.steps-300.clamp-no_clamp.txt.ctrl`. 

- Let's also generate 500 samples from the unguided model for comparison:

```sh
CUDA_VISIBLE_DEVICES=8 && bash scripts/text_sample.sh ckpts/simplev2/ema_0.9999_005001.pt 300 500
```

* Let's compare the outputs of the two models:

```sh
# top 5 colors in the unguided output:

(diffusion) amadaan@sa:~/home2/minimal-text-diffusion$ cut -f3 -d" " ckpts/simplev2/ema_0.9999_005001.pt.samples_500.steps-300.clamp-no_clamp.txt | sort | uniq -c | sed 's/^\s*//g' | sort -n|tail -5
30 purple
53 yellow
69 green
111 pink
166 white
```

```sh
# top 5 colors in the guided output:
(diffusion) amadaan@sa:~/home2/minimal-text-diffusion$ cut -f3 -d" " ckpts/simplev2/ema_0.9999_005001.pt.samples_500.steps-300.clamp-no_clamp.txt.ctrl.sample1 | sort | uniq -c | sed 's/^\s*//g' | sort -n|tail -5
15 pink
16 black
25 purple
124 yellow
269 green
```

* 50% of the sentences in the guided output contain the color word "green" vs. 69/500 = 14% in the unguided output. It looks like it's working! (recall that green was one of the 4 colors we specified in the classifier for label 1).



## Implementation Details

- The files relevant to controllable generation are in `src/controllable/`.

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

