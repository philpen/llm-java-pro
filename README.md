
# llm-java-pro

Implementation of the GPT2 Large Language Model (LLM) example in Java. This is a port of the original project (llm.c) found [here](https://github.com/karpathy/llm.c). 

## Preparation for Running ChatGPT2 in Java

It is necessary to undertake some preparatory steps before running. These steps mirror those found in the original [llm.c repository](https://github.com/karpathy/llm.c). Despite the presence of the same code in this repository, bear in mind that LLM.c is an ongoing project.

It is strongly recommended to run the original llm.c to gain perspective of its functioning.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
```

### JVM Requirements

This version runs on GraalVM 21. If you're using sdkman, it is necessary to use this command: