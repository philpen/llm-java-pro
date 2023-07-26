
package org.hjackson.llm;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.stream.IntStream;

public class GPT2 {
    private final AtomicBoolean activationsMem = new AtomicBoolean(false);
    private ExecutorService executorService = Executors.newFixedThreadPool(1000);
    private final AtomicLong gpt2_forward_counter = new AtomicLong();
    public final AtomicLong gpt2_forward_counter_layer = new AtomicLong();
    private final AtomicLong gpt2_backward_counter_layer = new AtomicLong();
    private final static AtomicLong input_counter = new AtomicLong();
    private final static AtomicLong target_counter = new AtomicLong();
    private static float GELU_SCALING_FACTOR = (float) Math.sqrt(2.0f / Math.PI);

    public final GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    public final ParameterTensors params;
    private int num_parameters;
    // gradients of the weights
    public ParameterTensors grads;
    private float grads_memory;
    // buffers for the AdamW optimizer
    private double[] m_memory;
    private double[] v_memory;
    // the activations of the model, and their sizes
    public ActivationTensors acts;
    private int num_activations;
    // gradients of the activations
    private ActivationTensors grads_acts;
    // other run state configuration
    private int batch_size; // the batch size (B) of current forward pass
    private int seq_len; // the sequence length (T) of current forward pass
    private DataLoader loader;
    //private DataLoader train_loader;
    public float mean_loss = 0.0f; // after a forward pass with targets, will be populated with the mean loss
    private final long file_size;
    private final Arena memoryArena;
    private final MemorySegment data;
    private static final int headerSize = 256 * 4;// bytes
    private final IntBuffer header = IntBuffer.allocate(256);
    private volatile boolean debugging = true;

    public GPT2(String checkpoint_path) throws Exception {
        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpoint_path),
                StandardOpenOption.READ)) {
            this.file_size = fileChannel.size();
            this.memoryArena = Arena.ofAuto();
            System.out.printf("File Size: %d\n", file_size);
            MemorySegment mappedFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, this.file_size, this.memoryArena);
            this.data = mappedFile;
            alloc_header(mappedFile, header);
            if (header.get(0) != 20240326) {
                throw new Exception("Bad version in model file");
            }
            if (header.get(1) != 3) {
                System.out.printf("Bad version in model file\n");
                System.out.printf("---> HINT: try to re-run `python train_gpt2.py`\n");
                System.exit(1);
            }
            config = new GPT2Config(header);
            assert (config.vocab_size > 0);
            params = new ParameterTensors(mappedFile, config);
            gpt2_build_from_checkpoint(checkpoint_path);
        }
        final float wte0 = params.mem[0];
        Assert.floatEquals(wte0, -0.11010301f);
    }

    void accessingInputs(String method, int b, int T, int t, int val) {
        long cnt = input_counter.incrementAndGet();
        if(cnt % 16 == 0) {
            System.out.printf("%s inputCounter %d val == %d\n", method, cnt, val);
        }
//        if(cnt == 100) {
//            System.out.printf("%s inputCounter %d val == %d %s\n", method, cnt, val, this);
//        }
    }

    void accessingTargets(String method, int b, int T, int t, int val) {
        long cnt = target_counter.incrementAndGet();
        if(cnt % 16 == 0) {
            System.out.printf("%s targetCounter == %d val == %d\n", method, cnt, val);
        }
//        if(cnt == 100) {
//            System.out.printf("%s targetCounter == %d val == %d %s\n", method, cnt, val, this);
//        }
    }

    public void stop() {
        if(debugging) {
            System.exit(1);
        }
    }

    public void gpt2_build_from_checkpoint(final String checkpoint_path) throws Exception {

        int maxT, V, Vp, L, NH, C;
        this.config.max_seq_len = maxT = header.get(2);
        this.config.vocab_size = V = header.get(3);
        this.config.num_layers = L = header.get(4);
        this.config.num_heads = NH = header.get(5);
        this.config.channels = C = header.get(6);
        this.config.padded_vocab_size = Vp = header.get(7);

        System.out.printf("[GPT-2]\n");
        System.out.printf("max_seq_len: %d\n", maxT);
        System.out.printf("vocab_size: %d\n", V);
        System.out.printf("padded_vocab_size: %d\n", Vp);
        System.out.printf("num_layers: %d\n", L);
        System.out.printf("num_heads: %d\n", NH);
        System.out.printf("channels: %d\n", C);
        System.out.printf("num_parameters: %d\n", params.getNumParams());
        this.num_parameters = params.getNumParams();

        // read in all the parameters from file
        // this.params_memory = malloc_and_point_parameters(this.params, 256, is);
        // other inits
        this.grads_memory = 0;
        this.m_memory = null;
        this.v_memory = null;
        this.batch_size = 0;
        this.seq_len = 0;
        this.mean_loss = -1.0f; // -1.0f will designate no loss
    }

    public void alloc_header(MemorySegment mappedFile, IntBuffer header) throws Exception {
        int startPos = 0;
        int endPos = headerSize;