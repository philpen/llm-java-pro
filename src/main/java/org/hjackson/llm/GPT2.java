
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
        IntBuffer tmp = mappedFile.asSlice(startPos, endPos).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        //tmp is a view into the mapped file so we need to copy it
        System.out.printf("intBuffer size: %d\n", tmp.capacity());
        header.put(tmp);
        System.out.printf("header[0]=%d\n", header.get(0));
        System.out.printf("header[1]=%d\n", header.get(1));
    }

    void gpt2_zero_grad() {
        if (grads_acts != null) {
            grads_acts.zeroFill();
        }
        if (grads != null) {
            grads.zeroFill();
        }
    }
    //        encoder_backward(grads.wte, grads.getWpe, grads_acts.getEncoded, loader, B, T, C);
    private void encoder_backward(int dwte, int dwpe, int dout, DataLoader inputs, int B, int T, int C) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;//grads_acts
                int ix = inputs.getInputs(b * T + t);
                //accessingInputs("encoder_backward", b, T, t, ix);
                int dwte_ix = dwte + ix * C;//grads
                int dwpe_t = dwpe + t * C;//grads
                for (int i = 0; i < C; i++) {
                    float d = grads_acts.mem[dout_bt + i];
                    grads.mem[dwte_ix + i] += d;
                    grads.mem[dwpe_t + i] += d;
                }
            }
        }
    }

    void gpt2_update(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        // lazily allocate the memory for m_memory and v_memory
        if (m_memory == null) {
            m_memory = new double[(int) num_parameters];
            v_memory = new double[(int) num_parameters];
        }
        for (int i = 0; i < num_parameters; i++) {
            double param = params.mem[i];
            double grad = grads.mem[i];
            // update the first moment (momentum)
            double m = beta1 * m_memory[i] + (1.0 - beta1) * grad;
            // update the second moment (RMSprop)
            double v = beta2 * v_memory[i] + (1.0 - beta2) * grad * grad;
            // bias-correct both moments
            double m_hat =  (m / (1.0 - Math.pow(beta1, t)));
            double v_hat =  (v / (1.0 - Math.pow(beta2, t)));
            // update
            m_memory[i] = m;
            v_memory[i] = v;
            params.mem[i] -= (float) (learning_rate * (m_hat / (Math.sqrt(v_hat) + eps) + weight_decay * param));
        }
    }

    //                    grads_acts, grads_acts, grads_acts, grads_acts, acts, acts
    void attention_backward(int dinp, int dpreatt, int datt, int dout, int inp, int att, int B, int T, int C, int NH) {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C*3;
        int hs = C / NH; // head size
        float scale = (float) (1.0f / Math.sqrt(hs));
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < NH; h++) {
                    int att_bth = att + b*NH*T*T + h*T*T + t*T; //acts
                    int datt_bth = datt + b*NH*T*T + h*T*T + t*T; //grads_acts
                    int dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;// grads_acts
                    int dquery_t = dinp + b * T * C3 + t * C3 + h * hs;//grads_acts
                    int query_t = inp + b * T * C3 + t * C3 + h * hs;//acts

                    // backward pass 4, through the value accumulation
                    int dout_bth = dout + b * T * C + t * C + h * hs;//grads_acts
                    for (int t2 = 0; t2 <= t; t2++) {
                        int value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value // acts.
                        int dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;//grads_acts
                        for (int i = 0; i < hs; i++) {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            grads_acts.mem[datt_bth + t2] += acts.mem[value_t2 + i] * grads_acts.mem[dout_bth + i];
                            grads_acts.mem[dvalue_t2 + i] += acts.mem[att_bth + t2] * grads_acts.mem[dout_bth + i];
                        }
                    }
                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0; t2 <= t; t2++) {
                        for (int t3 = 0; t3 <= t; t3++) {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            float local_derivative = acts.mem[att_bth + t2] * (indicator - acts.mem[att_bth + t3]);
                            grads_acts.mem[dpreatt_bth + t3] += local_derivative * grads_acts.mem[datt_bth + t2];
                        }
                    }
                    // backward pass 1, the query @ key matmul
                    for (int t2 = 0; t2 <= t; t2++) {
                        int key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key   // acts
                        int dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key // grads_acts
                        for (int i = 0; i < hs; i++) {
                            // in the forward pass this was:
                            grads_acts.mem[dquery_t + i] += acts.mem[key_t2 + i] * grads_acts.mem[dpreatt_bth + t2] * scale;
                            grads_acts.mem[dkey_t2 + i] += acts.mem[query_t + i] * grads_acts.mem[dpreatt_bth + t2] * scale;
                        }
                    }
                }
            }
        }
    }
    //                          grads_acts, acts,  grads_acts
    private void gelu_backward(int dinp, int inp, int dout, int N) {
        for (int i = 0; i < N; i++) {
            double x = acts.mem[inp + i];
            double cube = 0.044715 * x * x * x;
            double tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            double tanh_out = (float) Math.tanh(tanh_arg);
            double coshf_out = (float) Math.cosh(tanh_arg);
            double sech_out = 1.0 / (coshf_out * coshf_out);
            double local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
            grads_acts.mem[dinp + i] += (float) (local_grad * grads_acts.mem[dout + i]);
        }
    }
                          //       grads_acts, grads,     grads,    grads_acts, acts    , params   , acts    , acts
    //           layernorm_backward grads_acts, grads,     grads,    grads_acts, residual, params   , acts    , acts    , B, T, C);
    private void layernorm_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int mean, int rstd, int B, int T, int C) {

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dout_bt = dout + b * T * C + t * C;//grads_acts
                int inp_bt = inp + b * T * C + t * C;//acts
                int dinp_bt = dinp + b * T * C + t * C;//grads_acts