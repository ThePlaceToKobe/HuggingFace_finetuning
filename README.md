# HuggingFace finetuning

The goal is to use huggingface models and tokenizer in order to specify them on your own dataset / oriented new task.
To do so, we implemented several pipelines using differents models and evaluate them.

Here, we are able to finetune BERT-type model (such as bert-base, bert-base-multilingual, camembert-base, flaubert, etc.) and generative text-to-text models (such as t5-base, mt5, etc.)

BERT-type model can be easily finetune on classification task, so the evaluation is a classic evaluation pipeline.
Generative Text2Text are very different models as they generate directly text, so, we implement a Text2Classification procedure to transform the output text in predicted class.
