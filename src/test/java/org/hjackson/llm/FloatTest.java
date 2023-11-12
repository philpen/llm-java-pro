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
    public void floatTest