## Experiments and Results

* I've tried experiments with the following hyperparameters:

1. Embeddings/vocab: pre-trained `bert-base-uncased` vs. initialized randomly.

2. Model backbone: pre-trained `bert-base-uncased` vs. initialized from scratch.

3. Embeddings fine-tuning: fine-tuned vs. frozen.

Out of the 8 possible combinations, the best results were obtained with the following hyperparameters:

| File                                               | Sample Sentences                                                                                                                                                           | Perplexity         | % Unique Lines | % Unique Tokens |
|----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|----------------|-----------------|
| MODEL_PT-False_EMBEDS_PT-False-FREEZE_EMBEDS-False | The yellow lentil is stir-fried., The green lemon is stir-fried., The white kiwi is deep-fried., The orange turnip is stir-fried., The blue blackberry is deep-fried.      | 212.70 | 80.0           | 3.76            |
| MODEL_PT-False_EMBEDS_PT-False-FREEZE_EMBEDS-True  | The green spinach is stir-fried., The pink pomelo is stir-fried., The brown onion is stir-fried., The yellow artichoke is stir-fried., The blue pomegranate is deep-fried. | 218.77  | 74.2           | 3.76            |
| MODEL_PT-True_EMBEDS_PT-True-FREEZE_EMBEDS-False   | the yellow poccoli isggy., the red pe is sauteed., the green spinach is candied., the green danmelli isy., the brown kale is candied.                                      | 1424.21  | 78.0           | 6.1             |


---

- <s>The best setting in terms of diversity is using pre-trained bert, bert embeddings, and fine-tuning the embeddings. However, this setting has the lowest perplexity because it generates weird sentences (which could be a good thing depending on the application!).</s>

- Some random samples from <s>this setting</s>:

- Update 10/24: The following samples were likely from the `MODEL_PT-False_EMBEDS_PT-False-FREEZE_EMBEDS-False` setting. 

```
the purple chard is braised. 
the pink eggplant is soft. 
the blue perychmmon is fluffy. 
the pink eggplant is fluffy. 
the orange macadamia is spicy. 
the blue almond is poached. 
the black avychnd is steamed. 
the brown radicchio is delicious. 
the blue yam is microwaved. 
the black pistachio is dried. 
```

- The model was trained on a single RTX 2080 Ti GPU for 25k steps. The training time was ~1 hours.

---