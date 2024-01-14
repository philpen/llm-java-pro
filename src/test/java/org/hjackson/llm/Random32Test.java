package org.hjackson.llm;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.junit.jupiter.api.Assertions.*;

class Random32Test {
    @Test
    void random_f32() {
        Assertions.assertEquals(1337, Random32.RNG_STATE);
        Assertions.assertEquals(0.23031723499298096f, Random32.random_f32(Random32.RNG_STATE));
