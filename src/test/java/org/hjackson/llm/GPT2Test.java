
package org.hjackson.llm;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.time.Instant;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
class GPT2Test {
    float epsilon = 1e-2f;
    private int B;
    private int T;
    private GPT2 model;
    private long file_size;
    private Arena memoryArena;
    private MemorySegment mappedFile;
    private ParameterTensors expected_grads;
    private ByteBuffer byteBuffer;
    private int C;
    private int V;
    private int maxT;
    private int L;
    private int[] state_header;
    private byte[] mem;
    private int cpos;
    private int n;
    private int Vp;

    @BeforeEach
    void setUp() throws Exception {

        String stateFile = "gpt2_124M_debug_state.bin";
        String checkpoint_path = "gpt2_124M.bin";
        model = new GPT2(checkpoint_path);

        this.C = model.config.channels;
        this.V = model.config.vocab_size;
        this.Vp = model.config.padded_vocab_size;
        this.maxT = model.config.max_seq_len;
        this.L = model.config.num_layers;

        state_header = new int[256];
        RandomAccessFile state_file = new RandomAccessFile(new File(stateFile), "r");
        /**
         * Reading the entire file into mem is orders of magnitude faster than looping over the file and reading it in.
         */
        mem = new byte[(int) state_file.length()];
        cpos = 0;
        state_file.read(mem);
        byteBuffer = ByteBuffer.wrap(mem);
        state_file.close();
        for (int i = 0; i < 256; i++) {
            state_header[i] = byteBuffer.order(ByteOrder.nativeOrder()).getInt();
            cpos += 4;
            Assertions.assertEquals(cpos, byteBuffer.position());
        }
        Assertions.assertEquals(20240327, state_header[0], "Bad magic state file");
        Assertions.assertEquals(2, state_header[1], "Bad version in state file");
        B = this.state_header[2]; // batch size, e.g. 4
        T = this.state_header[3];
        System.out.printf("[State]\n");
        System.out.printf("batch_size: %d\n", B);
        System.out.printf("seq_len: %d\n", T);
    }

    //@Test
    void gpt2_build_from_checkpoint() throws IOException {
        Assertions.assertEquals(20240327, state_header[0], "Bad magic state file");
        Assertions.assertEquals(2, state_header[1], "Bad version in state file");
        int[] x = new int[B * T];
        int[] y = new int[B * T];
        for (int i = 0; i < x.length; i++) {
            x[i] = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
            cpos +=4;
            Assertions.assertEquals(cpos, byteBuffer.position());
            //System.out.printf("%d %d\n", n, x[i]);
        }
        DataLoader loader = new DataLoader(x, B, T, "test", true);
        for (int i = 0; i < y.length; i++) {
            y[i] = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getInt();
            cpos +=4;
            Assertions.assertEquals(cpos, byteBuffer.position());
            //System.out.printf("%d x[i] == %s y[i] == %s\n", i, x[i], y[i]);
        }
        float expected_loss = 0.0f;
        final int num_params = model.getNumParams();
        int btv = B * T * V;
        float[] expected_logits = new float[btv];
        System.out.printf("reading expected_logits\n");
        for (int i = 0; i < btv; i++) {
            float f = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getFloat();
            expected_logits[i] = f;
            cpos +=4;
            Assertions.assertEquals(cpos, byteBuffer.position());
        }

        expected_loss = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getFloat();
        cpos +=4;
        Assertions.assertEquals(cpos, byteBuffer.position());
        float[] expected_grads_memory = new float[num_params];
        System.out.printf("reading expected_grads_memory num_params == %d pos == %d rem == %d\n", num_params, cpos, byteBuffer.remaining());
        for (int i = 0; i < num_params; i++) {
            //System.out.printf("%d\n", i);
            float f = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getFloat();
            //System.out.printf("%d %1.17f\n", n, f);
            expected_grads_memory[i] = f;
            cpos +=4;
            Assertions.assertEquals(cpos, byteBuffer.position());
        }
        System.out.printf("cpos == %d\n", cpos);
        Assertions.assertEquals(cpos, byteBuffer.position());
        Assertions.assertEquals(549369860, byteBuffer.position());
    /* expected_logits[0]    == -43.43161774
       expected_loss         == 5.27000856
       expected_grads_memory == -0.00231974 */
        System.out.printf("expected_logits[0] == %f length == %d\n", expected_logits[0], expected_logits.length);
        System.out.printf("expected_loss            == %f\n", expected_loss);
        System.out.printf("expected_grads_memory[0] == %f length == %d\n", expected_grads_memory[0], expected_grads_memory.length);
        // overall OK signal for the test
        boolean allok = true;
        // expected losses are as follows, from Python
        float[] expected_losses = {
                5.270007133483887f,
                4.059706687927246f,
                3.3751230239868164f,
                2.8007826805114746f,
                2.315382242202759f,
                1.8490285873413086f,
                1.3946564197540283f,
                0.9991465210914612f,
                0.6240804195404053f,
                0.37651097774505615f
        };
        // let's do 10 training iterations, following the pytorch code
        float[] losses = new float[10];
        for (int step = 0; step < 10; step++) {
            Instant start = Instant.now();
            model.gpt2_forward(loader, B, T);
            model.gpt2_zero_grad();
            model.gpt2_backward();
            Instant end = Instant.now();
            System.out.printf("Duration: %d seconds\n", Duration.between(end, start).getSeconds());
            ActivationTensors acts = model.acts;
            if (step == 0) {
                // error checking at step 0 for reference activations/gradients
                // at this point, target should be equal to expected_logits, let's compare
                boolean logits_ok = true;
                int calculated_logits = acts.getLogits();
                float max_diff = 0.0f;
                for (int bt = 0; bt < B * T; bt++) {

                    for (int v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                        int i = bt * Vp + v; // linearized index, using Vp
                        if (i < 10) {
                            System.out.printf("%1.10f, %1.10f\n", expected_logits[i], model.acts.mem[calculated_logits + i]);
                        }
                        float diff = Math.abs(expected_logits[bt * V + v] - model.acts.mem[calculated_logits + i]);
                        max_diff = Math.max(max_diff, diff);
                        if (diff >= 1e-2f) {
                            System.out.printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                            System.out.printf("%1.10f %1.10f\n", expected_logits[bt * V + v], model.acts.mem[calculated_logits + i]);
                            logits_ok = false;
                            bt = B * T; // to break out of both loops
                            break;
                        }
                    }
                }
                if (!logits_ok) {
                    System.out.printf("Logits not ok, exiting\n");
                    System.exit(1);
                }
                System.out.printf("OK (LOGITS)\n");
                allok = allok && logits_ok;
                // compare the achieved loss