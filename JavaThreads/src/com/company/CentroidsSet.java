package com.company;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CentroidsSet {
    private int numberOfCentroids;
    private List<Centroid> centroids;

    public CentroidsSet(int k) {
        this.numberOfCentroids = k;
        this.centroids = new ArrayList<>();
    }

    public void initCentroids (PointsSet points) {
        Random r = new Random();
        List<Integer> randoms = new ArrayList<>();
        int count = 0;

        while(count < numberOfCentroids) {
            int rnd = r.nextInt(points.getSize());
            if (!randoms.contains(rnd)) {
                randoms.add(rnd);
                float x = points.getPoint(rnd).getX();
                float y = points.getPoint(rnd).getY();
                addCentroid(new Centroid(x,y,count));
                count++;
            }
        }
    }

    public void showCentroid() {
        System.out.println("CENTROIDS");
        for (int i=0; i<numberOfCentroids; i++) {
            System.out.println(centroids.get(i).getX() + " " + centroids.get(i).getY() + " " + centroids.get(i).getPointsNumber());
        }
    }

    public int getNumberOfCentroids() {
        return numberOfCentroids;
    }

    public void addCentroid(Centroid p) {
        centroids.add(p);
    }

    public Centroid getCentroid(int pos) {
        return centroids.get(pos);
    }
}
