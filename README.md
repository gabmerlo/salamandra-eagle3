# salamandra-eagle3
Accelerate Salamandra-7B-instruct-2606 with EAGLE-3 and compare with the 2B GPTQ draft model

## Data

To be able to use EAGLE-3 I need a dataset with questions I can use to generate with salamandra-7b some answers for my training data. In my case, since I'm interested in checking how EAGLE-3 accelerates Salamandra in languages from the iberian peninsula like catalan, basque and galician and spanish, I'm going to be using another resource from the BSC, in this case the ALIA-2606-SFT dataset:
https://huggingface.co/datasets/BSC-LT/ALIA-2606-SFT

In this dataset's card we can find a language field which I've used to filter the questions I want.

My next step will be to generate the answers to these questions with salamandra.