package com.company;

import java.util.concurrent.Callable;


public class UpdateCentroidThread implements Callable<Boolean> {
    private PointsSet points;
    private Centroid centr;

    public UpdateCentroidThread(PointsSet points, CentroidsSet c, int id) {
        this.points = points;
        this.centr = c.getCentroid(id);

    }

    @Override
    public Boolean call() {

        float x = 0;
        float y = 0;
        int count = 0;

        for (int i=0; i<points.getSize(); i++) {
            if (points.getPoint(i).getLabel() == centr.getId()) {
                count ++;
                x += points.getPoint(i).getX();
                y += points.getPoint(i).getY();
            }
        }
        x /= count;
        y /= count;
        centr.setX(x);
        centr.setY(y);
        centr.addPoints(count);
        return Boolean.TRUE;
    }
}
