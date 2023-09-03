
package org.hjackson.llm;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Llm {
    private static String tiny_stories_train = "data/TinyStories_train.bin";
    private static String tiny_stories_val = "data/TinyStories_val.bin";
    private static String tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    private static String tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";

    //private static final Random random = new Random(RNG_STATE);

    public static void main(String[] args) throws Exception {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        long xmx = memoryBean.getHeapMemoryUsage().getMax();
        System.out.printf("Max Mem %d\n", xmx);
        Llm llm = new Llm();
        if(xmx < 8589934590L) {
            throw new IllegalStateException("-Xmx needs to be at least -Xmx8192m");
        }
        if(!llm.getClass().desiredAssertionStatus()) {
            System.out.printf("Assertions are turned off, if editing the code I strongly recommend them to be on\n");
        }

        System.out.printf("Hello and welcome!\n");
        GPT2 model = new GPT2("gpt2_124M.bin");
        System.out.printf("wte[0] == %f\n", model.params.mem[0]);
        // build the DataLoaders from tokens files. for now use tiny_shakespeare if
        // available, else tiny_stories
        String train_tokens = Files.exists(Paths.get(tiny_shakespeare_train)) ? tiny_shakespeare_train : tiny_stories_train;
        String val_tokens = Files.exists(Paths.get(tiny_shakespeare_val)) ? tiny_shakespeare_val : tiny_stories_val;
        System.out.printf("Training with %s using values %s\n", train_tokens, val_tokens);
        final int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        final int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT,
        // which is 1024 for GPT-2
        DataLoader train_loader = new DataLoader(train_tokens, B, T, "train", true);
        System.out.printf("train dataset num_batches: %s\n", train_loader.num_batches);
        DataLoader val_loader = new DataLoader(val_tokens, B, T, "val", true);
        System.out.printf("val dataset num_batches: %s\n",  val_loader.num_batches);
        final int val_num_batches = 5;
        Tokenizer tokenizer = new Tokenizer("gpt2_tokenizer.bin");
        // some memory for generating samples from the model
        // int gen_max_length = 64;
        // during inference step we'll generate sequences of this many tokens
        final int[] gen_tokens = new int[B * T];
        final int genT = 64;
        // train
        long start, end;
        for(int step = 0; step <= 40; step++) {
            // once in a while estimate the validation loss
            if (step % 10 == 0) {
                float val_loss = 0.0f;
                val_loader.dataloader_reset();
                for (int i = 0; i < val_num_batches; i++) {
                    val_loader.dataloader_next_batch();
                    model.gpt2_forward(val_loader, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                System.out.printf("step %d val loss %f\n", step , val_loss);
            }
            // once in a while do model inference to print generated text
            if (step > 0 && step % 4 == 0) {
                // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
                for(int i = 0; i < B * T; ++i) {
                    gen_tokens[i] = tokenizer.end_of_text;
                }
                // now sample from the model autoregressively