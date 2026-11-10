# salamandra-eagle3

I wanted to see how much faster I could make Salamandra without touching its weights, so I trained my own EAGLE-3 draft model.

Salamandra-7B-Instruct-2606 is an LLM developed by the Barcelona Supercomputing Center (BSC), with special attention to Spanish, Catalan, Basque and Galician, four languages from the Iberian Peninsula.

For this experiment, I generated my own training dataset using prompts from ALIA-2606-SFT and regenerating the answers with Salamandra itself. I ended up with 30,973 records and trained the draft for 3 epochs on a single H100 80GB, taking around 23 minutes per epoch.

First result: Salamandra generates around 2x faster with my EAGLE-3 draft across all four languages (on average). 

I still want to compare it with a 2B GPTQ draft model and investigate why some answers change during speculative decoding.

<p align="center">
  <img src="graphs/mean_iberian_speedup.png" width="450" alt="Iberian Peninsula Languages Speedup">
</p>

<p align="center">
  <img src="graphs/speedup_languages.png" width="600" alt="Detailed Breakdown">
</p>

## Data

To be able to use EAGLE-3 I need a dataset with questions I can use to generate with salamandra-7b some answers for my training data. In my case, since I'm interested in checking how EAGLE-3 accelerates Salamandra in languages from the iberian peninsula like catalan, basque and galician and spanish, I'm going to be using another resource from the BSC, in this case the ALIA-2606-SFT dataset:
https://huggingface.co/datasets/BSC-LT/ALIA-2606-SFT

In this dataset's card we can find a language field which I've used to filter the questions I want.

I took 5000 conversations per language and regenerated the answers with salamandra itself using vLLM and the the regenerate-responses command from speculators, that way the draft learns from what salamandra really answers and not from the original dataset answers. I removed the 114 answers that got cut by max-tokens, so I ended up with 30,973 records. I also kept a separate set of 50 prompts per language that I never trained on, just for the evaluation.

## Training

I trained the draft with [speculators](https://github.com/vllm-project/speculators) in online mode on RunPod, with a single H100 80GB shared between vLLM (which gives the hidden states of salamandra) and the training. 3 epochs, around 23 minutes each, with a reduced draft vocabulary of 32000 tokens.

Some things failed on the way and I left them commented in the notebook, for example I had to prepare the data with sequences of 4000 tokens instead of 4096 because otherwise the training asked vLLM for 0 tokens and crashed.

In the validation of the last epoch the draft gets right the 1st, 2nd and 3rd proposed token in a row 66%, 44% and 30% of the times.

Checkpoints are on Hugging Face: https://huggingface.co/gabmerlo/salamandra-7b-eagle3-alia

## Results

(Awaiting an extra run comparing with 2B GPTQ draft model before writing this section)

