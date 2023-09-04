
package org.hjackson.llm;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.*;

public class ParameterTensors {
    public final int wte;
    private final int wte_size;
    private final int wpe;
    private final int wpe_size;
    private final int ln1w;
    private final int ln1w_size;
    private final int ln1b;
    private final int ln1b_size;
    private final int qkvw;
    private final int qkvw_size;
    private final int qkvb;
    private final int qkvb_size;
    private final int attprojw;
    private final int attprojw_size;
    private final int attprojb;
    private final int attprojb_size;
    private final int ln2w;
    private final int ln2w_size;
    private final int ln2b;
    private final int ln2b_size;
    private final int fcw;
    private final int fcw_size;
    private final int fcb;
    private final int fcb_size;
    private final int fcprojw;
    private final int fcprojw_size;
    private final int fcprojb;
    private final int fcprojb_size;
    private final int lnfw;
    private final int lnfw_size;
    private final int lnfb;
    private final int lnfb_size;
    private final int num_params;
    public final float[] mem;
    private boolean ok = false;
    private final Map<Integer, Float> tracking = new HashMap<>();
    public ParameterTensors(MemorySegment segment, GPT2Config config) {
        this(config);
        int pos = 1024;//header
        for (int i = 0; i < num_params; i++, pos += 4) {
            mem[i] = segment.get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.nativeOrder()), pos);
        }
        Assert.floatEquals(this.mem[0], -0.11010301f);
        runParamAssertions();
        ok = true;
    }
    public ParameterTensors(GPT2Config config) {
        int maxT = config.max_seq_len;
        int C = config.channels;
        int V = config.vocab_size;
        int Vp = config.padded_vocab_size;
        int L = config.num_layers;
        // allocate space for all the parameters and read them in
        wte_size = Vp * C;
        wte = 0;
        wpe_size = maxT * C;
        wpe = wte + wte_size;
        ln1w_size = L * C;
        ln1w = wpe + wpe_size;
        ln1b_size = L * C;
        ln1b = ln1w + ln1w_size;
        qkvw_size = L * (3 * C) * C;
        qkvw = ln1b + ln1b_size;
        qkvb_size = L * (3 * C);
        qkvb = qkvw + qkvw_size;
        attprojw_size = L * C * C;
        attprojw = qkvb + qkvb_size;
        attprojb_size = L * C;
        attprojb = attprojw + attprojw_size;