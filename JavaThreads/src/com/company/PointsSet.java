package com.company;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class PointsSet {
    private int size;
    private List<Point> points;

    public PointsSet() {
        this.size = 0;
        this.points = new ArrayList<>();
    }

    public void initPoints(String path) throws Exception {
        File file = new File(path);
        Scanner sc = new Scanner(file);
        String[] coordinates;

        while (sc.hasNextLine()) {
            String s = sc.nextLine();
            coordinates = s.split(",");
            float x = Float.parseFloat(coordinates[0]);
            float y = Float.parseFloat(coordinates[1]);
            Point p = new Point(x,y);
            addPoint(p);
        }
    }

    public int getSize() {
        return size;
    }

    public void addPoint(Point p) {
        points.add(p);
        size ++;
    }

    public Point getPoint(int pos) {
        return points.get(pos);
    }

    public void showPoints() {
        System.out.println("POINTS");
        for (int i=0; i<size; i++) {
            System.out.println(points.get(i).getX() + " " + points.get(i).getY() + " " + points.get(i).getLabel());
        }
    }
}
