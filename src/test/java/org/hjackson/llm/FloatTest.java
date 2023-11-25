package org.hjackson.llm;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class FloatTest {

    @Test
    public void randomTest() {
        float test = Random32.random_f32(1337l);
        System.out.printf("%1.17f", test);
        Assertions.assertEquals(0.23031723499298096f, test);
    }

    @Test
    public void floatTest() {
        float[] flts = new float[]{
                0.76013361582701647f,
                0.70070299091981488f,
                0.16198650599372622f,
                0.92492271720057188f,
                0.79394704958673160f,
                0.741609501681