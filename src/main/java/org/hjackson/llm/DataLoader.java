
package org.hjackson.llm;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.atomic.AtomicLong;

public final class DataLoader {
    private final int BT1;
    // hyperparameters
    public final int B; // batch size
    public final int T; // sequence length
    private final String type;
    // input handling and its state
    public RandomAccessFile tokens_file;
    public long file_size;
    public long current_position;
    // output memory
    public final int[] batch;
    private final int[] cache;
    private boolean workOnCache = false;
    // convenience variables
    public int num_batches;
    private boolean targetsPresent  = false;
    public DataLoader(String filename, final int B, final int T, String type, boolean targetsPresent) throws Exception {
        this(B, T, type);
        this.targetsPresent = true;
        // open the input file for reading
        this.tokens_file = new RandomAccessFile(new File(filename), "r");
        this.file_size = this.tokens_file.length();
        if (this.file_size < (B * T + 1) * 4) {
            throw new Exception("Error: file size is too small for the batch size and sequence length");
        }
        this.current_position = 0; // start at the beginning
        this.num_batches = (int) ( this.file_size / (B * T * 4));
    }
