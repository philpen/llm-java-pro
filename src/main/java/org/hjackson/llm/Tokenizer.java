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
    private RandomAccessFile file;
    private String fileName;
    private final int[] header = new int[256];
    //private final int init_ok;

    public Tokenizer(String filename) throws IOException {
        fileName = filename;

        File f = new File(filename);

        if (!f.exists()) {
            // try to be more helpful as we just a