package com.pengxl.welding;

public class MathFunction {
    public static float sigmoid(float value) {
        return (float) (1 / (1 + (Math.exp(Double.parseDouble(String.valueOf(-value))))));
    }
}
