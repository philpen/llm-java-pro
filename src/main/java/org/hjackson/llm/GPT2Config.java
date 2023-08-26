package org.hjackson.llm;
import java.nio.IntBuffer;
public final class GPT2Config {
    public int max_seq_len; // max sequence length, e.g. 1024
    public int vocab_size; // vocab size, e.g. 50257
    public int padded_vocab_size; // padded to e.g. %128==0, 50304
    public int