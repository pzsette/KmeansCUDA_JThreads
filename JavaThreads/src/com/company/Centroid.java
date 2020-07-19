package com.company;

public class Centroid {
    private float x;
    private float y;
    private int points;
    private int id;

    Centroid(float x, float y, int id) {
        this.x = x;
        this.y = y;
        this.points = 0;
        this.id = id;
    }

    public void addPoints(int p) {
        this.points = p;
    }

    public int getId() {
        return id;
    }

    public int getPointsNumber() { return this.points;}

    public float getX() {
        return x;
    }

    public void setX(float x) {
        this.x = x;
    }

    public float getY() {
        return y;
    }

    public void setY(float y) {
        this.y = y;
    }
}
