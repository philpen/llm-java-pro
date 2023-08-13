
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
                float mean_bt = acts.mem[mean + b * T + t];//acts
                float rstd_bt = acts.mem[rstd + b * T + t];//acts
                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++) {
                    //float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                    float norm_bti = (acts.mem[inp_bt + i] - mean_bt) * rstd_bt;
                    float dnorm_i = params.mem[weight + i] * grads_acts.mem[dout_bt + i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / C;
                dnorm_norm_mean = dnorm_norm_mean / C;
                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++) {
                    float norm_bti = (acts.mem[inp_bt + i] - mean_bt) * rstd_bt;
                    float dnorm_i = params.mem[weight + i] * grads_acts.mem[dout_bt + i];
                    // gradient contribution to bias
                    grads.mem[dbias + i] += grads_acts.mem[dout_bt + i];
                    grads.mem[dweight + i] += norm_bti * grads_acts.mem[dout_bt + i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    grads_acts.mem[dinp_bt + i] += dval;
                }
            }
        }
    }
    //                        grads_acts , grads   ,    grads    , grads_acts, acts,   params
    //                        grads_acts , grads     ,  MIN_VALUE, grads_acts, acts,   params
         //matmul_backward(grads_acts.getLnf, grads.getWte, IMIN_VALUE, grads_acts.getLogits, acts.getLnf(), params.wte, B, T, C, Vp);
    private void matmul_backward(int dinp, int dweight, int dbias, int dout, int inp, int weight, int B, int T, int C, int OC, int id) {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy
        // backward into inp first, parallelize over B,T
        //#pragma omp parallel for collapse(2)
        final int btMax = B * T;
        IntStream.range(0, btMax) // This is probably not the fastest way to thread this loop
                .parallel()
                .forEach(bt -> {
                    int b = bt / T;
                    int t = bt % T;

                    final int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                    final int dinp_bt = dinp + b * T * C + t * C;//grads_acts
                    for (int o = 0; o < OC; o++) {
                        int wrow = weight + o*C;//params
                        float d = grads_acts.mem[dout_bt + o];
                        for (int i = 0; i < C; i++) {
                            grads_acts.mem[dinp_bt + i] += params.mem[wrow + i] * d;
                        }
                    }
                });
        // backward into weight/bias, parallelize over output channels OC
        //#pragma omp parallel for
        IntStream.range(0, OC) // This is probably not the fastest way to thread this loop
                .parallel()
                .forEach(o -> {
                    for (int b = 0; b < B; b++) {
                        for (int t = 0; t < T; t++) {
                            int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                            int inp_bt = inp + b * T * C + t * C;//acts
                            int dwrow = dweight + o*C;//grads
                            float d = grads_acts.mem[dout_bt + o];
                            if (dbias != Integer.MIN_VALUE) {
                                grads.mem[dbias + o] += d;
                            }
                            for (int i = 0; i < C; i++) {
                                grads.mem[dwrow + i] += acts.mem[inp_bt + i] * d;
                            }
                        }
                    }
                });
    }

    //                        grads_acts , grads   ,    grads    , grads_acts, acts,   params
    //                        grads_acts , grads     ,  MIN_VALUE, grads_acts, acts,   params
    private void matmul_backward2(int dinp, int dweight, int dbias, int dout, int inp, int weight,
                                      int B, int T, int C, int OC, int id) {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy
        // backward into inp first, parallelize over B,T
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                final int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                final int dinp_bt = dinp + b * T * C + t * C;//grads_acts
                for (int o = 0; o < OC; o++) {
                    int wrow = weight + o * C;//params
                    float d = grads_acts.mem[dout_bt + o];

                    for (int i = 0; i < C; i++) {
                        grads_acts.mem[dinp_bt + i] += params.mem[wrow + i] * d;
                    }
                }
            }
        }
        // backward into weight/bias, parallelize over output channels OC
        for (int o = 0; o < OC; o++) {
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int dout_bt = dout + b * T * OC + t * OC;//grads_acts
                    int inp_bt = inp + b * T * C + t * C;//acts
                    int dwrow = dweight + o * C;//grads
                    float d = grads_acts.mem[dout_bt + o];
                    if (dbias != Integer.MIN_VALUE) {
                        float v = grads.mem[dbias + o] + d;//grads
                        grads.mem[dbias + o] = v;
                    }
                    for (int i = 0; i < C; i++) {
//                        if(o == 2 && b == 2 && t == 2 && i == 2) {
//                            System.out.printf("%d matmul_backward b==%d t==%d i==%d d==%f inp_bt==%f dwrow==%f dwrow_cell=%d\n",
//                                    id, b, t, i, d, acts.mem[inp_bt + i], grads.mem[dwrow + i], dwrow + i);
//                            grads.didChange(b + "-" + t + "-" + o + "-" + i);
//                            grads_acts.didChange(b + "-" + t + "-" + o + "-" + i);
//                        }
                        grads.mem[dwrow + i] += acts.mem[inp_bt + i] * d;
                    }
                }
            }
        }
    }

               //crossentropy_softmax_backward(grads_acts.getLogits, grads_acts.getLosses, acts.getProbs(), loader, B, T, V, Vp);
    private void crossentropy_softmax_backward(int dlogits, int dlosses, int probs, DataLoader targets, int B, int T, int V, int Vp) {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int dlogits_bt = dlogits + b * T * Vp + t * Vp;
                int probs_bt = probs + b * T * Vp + t * Vp;
                float dloss = grads_acts.mem[dlosses + b * T + t];
                int ix = targets.getTargets(b * T + t);
                //accessingTargets("crossentropy_softmax_backward", b, T, t, ix);
                // note we only loop to V, leaving the padded dimensions
                // of dlogits untouched, so gradient there stays at zero
                for (int i = 0; i < V; i++) {
                    float p = acts.mem[probs_bt + i];
                    float indicator = i == ix ? 1.0f : 0.0f;
                    grads_acts.mem[dlogits_bt + i] += (p - indicator) * dloss;
                }
            }
        }
    }

    public int sample_mult(int probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += acts.mem[probabilities + i];
            if (coin < cdf) {
                //System.out.printf("cdf == %f  probi == %f\n", cdf, acts.mem[probabilities + i]);
                return i;
            }
        }
        //System.out.printf("ROUNDING ERROR cdf == %f n - 1 == %d\n", cdf, n - 1);
        return n - 1; // in case of rounding errors
    }

    public void gpt2_forward(DataLoader loader, final int B, final int T) {
        gpt2_forward_counter.incrementAndGet();
        this.loader = loader;
        // ensure the model was initialized or error out
        if (!params.ok()) {
            throw new IllegalStateException("Error: model was not initialized properly.\n");
        }
        // convenience parameters
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        int NH = config.num_heads;
        int C = config.channels;
        // validate inputs, all indices must be in the range [0, V)
        for (int i = 0; i < B * T; i++) {
            assert (0 <= loader.getInputs(i) && loader.getInputs(i) < V);
            if (loader.targetsPresent()) {
                assert (0 <= loader.getTargets(i) && loader.getTargets(i) < V);
            }
        }
        // allocate space for all the activations if needed (done here, lazily)
        if (!activationsMem.get()) {
            activationsMem.set(true);
            batch_size = B;
            seq_len = T;
            acts = new ActivationTensors(config, B, T);
            if (gpt2_forward_counter.get() == 1L) {
                Assert.floatEquals(acts.mem[acts.getResidual3()], 0.0f);
            }
            num_activations = acts.getNumActivations();
            System.out.printf("num_activations: %d\n", num_activations);
            // also create memory for caching inputs and targets
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            // in principle we could get more clever here in the future, for now this is safest
            if (B != batch_size || T != seq_len) {
                System.out.printf("Model: B={} T={}, Desired: B={} T={}\n", batch_size, seq_len, B, T);
                System.exit(1);
            }
        }
        if (gpt2_forward_counter.get() == 1L) {
            Assert.floatEquals(acts.mem[acts.getResidual3()], 0.0f);
        }
        loader.cacheInputs();
        int residual;
        encoder_forward(acts.getEncoded(), loader, params.getWte(), params.getWpe(), B, T, C);// encoding goes into residual[0]
        for (int l = 0; l < L; l++) {
            long layerCount = gpt2_forward_counter_layer.incrementAndGet();
            //residual = l == 0 ? acts.encoded : acts.residual3[] + (l-1) * B * T * C;
            if (l == 0) {
                residual = acts.getEncoded();
            } else {
                residual = acts.getResidual3() + (l - 1) * B * T * C;
            }
            //System.out.printf("f==%d b==%d residual == %f\n", layerCount, gpt2_backward_counter_layer.get(), acts.mem[residual]);
            if (gpt2_forward_counter.get() == 1L && l == 0) {
                Assert.floatEquals(acts.mem[acts.getResidual3()], 0.0f);
            }
            // get the pointers of the weights for this layer
            int l_ln1w = params.getLn1w() + l * C;
            int l_ln1b = params.getLn1b() + l * C;
            int l_qkvw = params.getQkvw() + l * 3 * C * C;
            int l_qkvb = params.getQkvb() + l * 3 * C;
            int l_attprojw = params.getAttprojw() + l * C * C;
            int l_attprojb = params.getAttprojb() + l * C;
            int l_ln2w = params.getLn2w() + l * C;
            int l_ln2b = params.getLn2b() + l * C;
            int l_fcw = params.getFcw() + l * 4 * C * C;
            int l_fcb = params.getFcb() + l * 4 * C;
            int l_fcprojw = params.getFcprojw() + l * C * 4 * C;
            int l_fcprojb = params.getFcprojb() + l * C;
            // get the pointers of the activations for this layer
            int l_ln1 = acts.getLn1() + l * B * T * C;
            int l_ln1_mean = acts.getLn1Mean() + l * B * T;
            int l_ln1_rstd = acts.getLn1Rstd() + l * B * T;
            int l_qkv = acts.getQkv() + l * B * T * 3 * C;
            int l_atty = acts.getAtty() + l * B * T * C;
            int l_preatt = acts.getPreatt() + l * B * NH * T * T;
            int l_att = acts.getAtt() + l * B * NH * T * T;
            int l_attproj = acts.getAttproj() + l * B * T * C;
            int l_residual2 = acts.getResidual2() + l * B * T * C;
            int l_ln2 = acts.getLn2() + l * B * T * C;
            int l_ln2_mean = acts.getLn2Mean() + l * B * T;
            int l_ln2_rstd = acts.getLn2Rstd() + l * B * T;
            int l_fch = acts.getFch() + l * B * T * 4 * C;
            int l_fch_gelu = acts.getFchGelu() + l * B * T * 4 * C;
            int l_fcproj = acts.getFcproj() + l * B * T * C;
            int l_residual3 = acts.getResidual3() + l * B * T * C;

            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);//checked 1
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);//checked 1
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);//checked 1
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);//checked 1
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);//checked 1
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.getResidual3() + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.getLnf(), acts.getLnfMean(), acts.getLnfRstd(), residual, params.getLnfw(), params.getLnfb(), B, T, C);
        matmul_forward(acts.getLogits(), acts.getLnf(), params.getWte(), Integer.MIN_VALUE, B, T, C, Vp);
        softmax_forward(acts.getProbs(), acts.getLogits(), B, T, V, Vp);
        // also forward the cross-entropy loss function if we have the targets
        if (loader.targetsPresent()) {
            //System.out.printf("targets present\n");