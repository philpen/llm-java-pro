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
            // try to be more helpful as we just added this feature, erase later
            System.out.printf(
                    """
                                  ---
                                  WARNING: Failed to open the tokenizer file {}
                                           The Tokenizer is a new feature added April 14 2024.\n
                                           "Re-run `python train_gpt2.py` to write it\\n");
                                  ---       
                            """, filename);
        }
        file = new RandomAccessFile(f, "r");
        // read in the header
        for (int i = 0; i < 256; i++) {
            //file has integers in little endian, jvm is bigendian
            header[i] = Integer.reverseBytes(file.readInt());
        }
        System.out.printf("header[0] == %d\n", header[0]);
        System.out.printf("header[0] == %d\n", header[2]);
        assert (header[0] == 20240328);
        assert (header[1] == 2);
        vocab_size = header[2];
        end_of_text = header[3];
        if(end_of_text != 50256) {
            throw new UnexpectedException("Something ha