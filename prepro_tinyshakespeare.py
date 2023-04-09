
"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created data/ folder.
The script prints:

Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespeare_train.bin

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import requests
from tqdm import tqdm

import tiktoken
import numpy as np

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,