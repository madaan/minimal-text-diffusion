# Minimal text diffusion

- This repo is ideal for people who want to experiment with diffusion models in text-generation setting.

- The idea is to train a text-generation model from a text-file.

- The code has been repurposed from https://github.com/XiangLi1999/Diffusion-LM, which in turns borrows from a number of repos including: https://github.com/openai/glide-text2im

## Trimmed Text Diffusion

- A repo for experimenting with conditional text diffusion models that has only the bare minimum.


# Text-preprocessing
- Specify a text dataset over which you want to train a diffusion model.



- For this simple exercise, we will generate quotes sourced from https://gist.github.com/erickedji/68802, http://rvelthuis.de/zips/quotes.txt


https://raw.githubusercontent.com/skolakoda/programming-quotes-api/master/Data/quotes.json


- You start from creating raw embeddings from the given dataset.
