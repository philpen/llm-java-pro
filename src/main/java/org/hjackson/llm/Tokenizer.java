package org.hjackson.llm;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.rmi.UnexpectedException;

public class Tokenizer {
    private final int vocab_size;
    private final String[] token_table;
    public final int init_ok;
    final int end_of_text;
    private RandomAccessF