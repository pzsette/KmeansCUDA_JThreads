package com.company;

import java.util.concurrent.Callable;

public class AssignLabelThread implements Callable<Integer> {
    private PointsSet points;
    private CentroidsSet centroids;
    private int start;
    private int end;

    public AssignLabelThread(PointsSet p, CentroidsSet c, int start, int end) {
        this.points = p;
        this.centroids = c;
        this.start = start;
        this.end = end;
    }

    @Override
    public Integer call() {
        int updates = 0;
        for (int i=start; i<end; i++) {
            Point p = points.getPoint(i);
            float minDistance = Distance.computeDistance(p.getX(), p.getY(), centroids.getCentroid(0).getX(), centroids.getCentroid(0).getY());
            int minIndex = 0;
            for (int j=1; j<centroids.getNumberOfCentroids(); j++) {
                float distance = Distance.computeDistance(p.getX(), p.getY(), centroids.getCentroid(j).getX(), centroids.getCentroid(j).getY());
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = j;
                }
            }
            if (p.getLabel() != minIndex) {
                p.setLabel(minIndex);
                updates++;
            }
        }
        return updates;
    }
}
