package NeuralNetwork;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by Andre on 13/05/2017.
 */
public class Util {

    public static final int NUM_DECIMALS = 10;

    public static double[][] transposeMatrix(double[][] m) {
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;

    }

    public static double[][] multiply(double[][] A, double[][] B, double[][] C) {

        int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = B[0].length;

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }

        for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                for (int k = 0; k < aColumns; k++) { // aColumn
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    public static double[][] calculateNeuronOutput(double[][] A, double[][] B, boolean isOutputLayer) {

        int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = B[0].length;

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }

        double[][] C;
        if(isOutputLayer){
            C = new double[aRows][bColumns];
        }else{
            C = new double[aRows][bColumns+1];
            for (int i = 0; i < aRows; i++) {
                C[i][bColumns] = 1.0;
            }
        }

        C = multiply(A,B,C);

        for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                C[i][j] = sigmoid(C[i][j]);
            }
        }

        return C;
    }

    public static double sigmoid(double x){
        return (1 / (1 + Math.exp(-x)));
    }

//    public static double round(double value, int places) {
//        if (places < 0) throw new IllegalArgumentException();
//        return new BigDecimal(value).setScale(places, RoundingMode.HALF_UP).doubleValue();
//    }


}