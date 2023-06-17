package org.hjackson.llm;
public final class Assert {
    public static final float EPSILON = 1e-6f;
    public static void floatEquals(float a, float b) {
        float abs = Math.abs(a - b);
        if(!nearlyEqual(a, b, EPSILON)) {
            throw new IllegalStateException("float diff too big " + abs);
        }
        if(abs > EPSILON) {
            throw new IllegalStateException("float diff too big " + abs);
        }
    }
    public static void nonNan(final float meanLoss) {
        if(Float.isNaN(meanLoss)) {
            throw new IllegalStateException("NaN Found");
        }
    }
    public static boolean nearlyEqual(float a, float b, float epsilon) {
        final float absA = Math.abs(a);
        final float absB = Math.abs(b);
        final float diff = Mat