package org.hjackson.llm;
public final class Assert {
    public static final float EPSILON = 1e-6f;
    public static void floatEquals(float a, float b) {
        float abs = Math.abs(