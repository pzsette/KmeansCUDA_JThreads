package com.company;
import java.io.File;  // Import the File class
import java.io.FileWriter;
import java.io.IOException;  // Import the IOException class to handle errors

public class WriteResults {

    static private void createResFile () {
        try {
            File myObj = new File("results.txt");
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    static void writeResFile(PointsSet points) {
        createResFile();
        emptyFile();
        try {
            FileWriter f0 = new FileWriter("results.txt");

            String newLine = System.getProperty("line.separator");

            for(int i=0;i<points.getSize();i++)
            {
                Point pointToWrite = points.getPoint(i);
                f0.write(pointToWrite.getX()+","+pointToWrite.getY()+","+pointToWrite.getLabel()+newLine);
            }
            f0.close();

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    static private void emptyFile() {

        try {
            FileWriter f0 = new FileWriter("results.txt");

            f0.write("");

        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
