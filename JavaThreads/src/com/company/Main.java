package com.company;


import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {

    public static void main(String[] args) {

        PointsSet points = new PointsSet();
        try {
            points.initPoints("points.txt");
        } catch (Exception e) {
            System.out.println("Error initializing point");
        }

        CentroidsSet centroids = new CentroidsSet(5);
        centroids.initCentroids(points);

        Kmeans kmeans = new Kmeans(points, centroids);

        //Parallel execution
        int numCore = 4;
        ExecutorService executor = Executors.newFixedThreadPool(numCore);
        kmeans.parallelExecution(executor, numCore);
        executor.shutdown();

        //Sequential execution
        //kmeans.sequentialExecution();

        //centroids.showCentroid();
        //points.showPoints();

        //Create a "results.txt" file with all labeled points
        WriteResults.writeResFile(points);
    }
}
