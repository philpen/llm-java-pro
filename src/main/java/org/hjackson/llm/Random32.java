package org.hjackson.llm;

import java.math.BigInteger;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class Random32 {
    public static Long RNG_STATE = Long.parseUnsignedLong("1337");

    private static long random_u32(Long state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >>> 12;
        state ^= state << 25;
        state ^= state >>> 27;
        RNG_STATE = state;

        Bi