
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