package com.company;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class Kmeans {
    private PointsSet points;
    private CentroidsSet centroids;

    public Kmeans(PointsSet p, CentroidsSet c) {
        this.points = p;
        this.centroids = c;
    }

    //Parallel K-means Execution
    public void parallelExecution(ExecutorService executor, int threadNumber) {
        int updates;

        long startTime = System.currentTimeMillis();

        do {
            updates = 0;

            //Updating Points Label
            int task_size = points.getSize() / threadNumber + ((points.getSize() % threadNumber == 0) ? 0 : 1);

            List<AssignLabelThread> tasks = new ArrayList<>();
            for (int j = 0; j < threadNumber; j++) {
                tasks.add(new AssignLabelThread(points, centroids, j * task_size, (j + 1) * task_size));
            }

            try {
                List<Future<Integer>> numberOfUpdates = executor.invokeAll(tasks);

                for (Future<Integer> x : numberOfUpdates) {
                    updates += x.get();
                }

            } catch (InterruptedException | ExecutionException e) {
                System.out.println("error: "+e.getMessage());
            }

            //Updating Centroids
            List<UpdateCentroidThread> centroidsTasks = new ArrayList<>();
            for (int i=0; i<centroids.getNumberOfCentroids(); i++) {
                centroidsTasks.add(new UpdateCentroidThread(points, centroids, i));
            }

            try {
                executor.invokeAll(centroidsTasks);
            } catch (InterruptedException e) {

                System.out.println("error: "+e.getMessage());
            }

            //System.out.println(updates + " updates done");

        } while (updates>0);

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\nParallel execution time -> "+estimatedTime+"ms\n");
    }

    //Sequential K-means Execution
    public void sequentialExecution() {
        int updates;

        long startTime = System.currentTimeMillis();

        do {
            updates=0;

            for (int i = 0; i < points.getSize(); i++) {

                Point p = points.getPoint(i);
                float minDistance = Distance.computeDistance(p.getX(), p.getY(), centroids.getCentroid(0).getX(), centroids.getCentroid(0).getY());
                int minIndex = 0;

                for (int j = 1; j < centroids.getNumberOfCentroids(); j++) {
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

            for (int k = 0; k < centroids.getNumberOfCentroids(); k++) {
                Centroid currentCentroid = centroids.getCentroid(k);

                float x = 0;
                float y = 0;
                int count = 0;

                for (int i = 0; i < points.getSize(); i++) {

                    if (points.getPoint(i).getLabel() == currentCentroid.getId()) {
                        count++;
                        x += points.getPoint(i).getX();
                        y += points.getPoint(i).getY();
                    }
                }
                if (count!=0) {
                    x /= count;
                    y /= count;
                }

                currentCentroid.setX(x);
                currentCentroid.setY(y);
                currentCentroid.addPoints(count);
            }

            //System.out.println(updates + " updates done");

        } while (updates>0);

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\nSequential execution time -> "+estimatedTime+"ms\n");
    }
 }
