# TinyStories

Minimal reproduction of GPT-NeoX, based on the [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L297).

Used with the existing [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories/viewer/roneneldan--TinyStories/validation) to train a model that can generate short stories, along with interperetation utilities.

Might need to run
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
```
before starting.

### Next steps

- [] Fine tune on fantasy dataset