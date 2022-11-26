# HuggingFace finetuning

The goal is to use huggingface models and tokenizer in order to specify them on your own dataset / oriented new task.
To do so, we implemented several pipelines using differents models and evaluate them.


| model family | examples of models | model explanation |
|:----:|:----:|:----:|
| Zero-shot learning | distilcamembert-base-nli ; xlm-roberta-large-xnli | These models don't need to be trained, use them just in inference |
| SkLearn models | RandomForestClassifier, SVClassifier | Classic Machine Learning interpretable models |
| BERT | camembert-base, bert-multilingual | Pretrained models which need to be finetune on your dataset and your task |
| Generative Text2Text | t5-base, mt5-large | Models which trained on text and directly generate text as an output |

Here, we are able to finetune BERT-type model (such as bert-base, bert-base-multilingual, camembert-base, flaubert, etc.) and generative text-to-text models (such as t5-base, mt5, etc.)

BERT-type model can be easily finetune on classification task, so the evaluation is a classic evaluation pipeline.
Generative Text2Text are very different models as they generate directly text, so, we implement a Text2Classification procedure to transform the output text in predicted class.
