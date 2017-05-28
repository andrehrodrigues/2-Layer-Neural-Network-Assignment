package NeuralNetwork;

import jsc.distributions.Uniform;

import java.util.Random;


/**
 * Created by Andre on 13/05/2017.
 */
public class NeuronLayer {

    int size;
    int previousLayerSize;
    double[][] weights;
    double[][] previousWeights;

    NeuronLayer(int size){
        this.size = size;
    }

    public NeuronLayer(int size, int previousLayerSize){
        this.size = size;
        this.previousLayerSize = previousLayerSize;
        this.weights = new double[size][previousLayerSize+1];

        setInitialWeights(size, previousLayerSize);
    }

    public void setInitialWeights(){
        Random r = new Random();
        for(int a = 0; a < size; a++){
            for (int b = 0; b < previousLayerSize+1; b++) {
                this.weights[a][b] = r.nextDouble();
            }
        }
    }

    public void setInitialWeights(int numInputs, int numOutputs){
        double r =  4*( Math.sqrt( (6.0/(Double.valueOf(numInputs+numOutputs).doubleValue())) ) ) ;

        Uniform uni = new Uniform( -r, r );
        uni.setSeed(System.currentTimeMillis());

        for(int a = 0; a < size; a++){
            for (int b = 0; b < previousLayerSize+1; b++) {
                this.weights[a][b] = uni.random();
            }
        }

    }

    public double[][] getLayerResult(double[][] input){
        double[][] hiddenLayerTranspose = Util.transposeMatrix(this.weights);

        double[][] resultLayer = Util.calculateNeuronOutput(
                input, hiddenLayerTranspose, false);

        return resultLayer;
    }

    public int getPreviousLayerSize() {
        return previousLayerSize;
    }

    public void setPreviousLayerSize(int previousLayerSize) {
        this.previousLayerSize = previousLayerSize;
    }

    public int getSize() {
        return size;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public double[][] getPreviousWeights() {
        return previousWeights;
    }

    public void setPreviousWeights(double[][] previousWeights) {
        this.previousWeights = previousWeights;
    }
}
