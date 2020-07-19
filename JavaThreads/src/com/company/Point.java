package com.company;

public class Point {
    private float x;
    private float y;
    private int label;

    public Point(float x, float y) {
        this.x = x;
        this.y = y;
        this.label = -1;
    }

    public float getX() {
        return x;
    }

    public float getY() {
        return y;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }
}
