package NeuralNetwork;

import java.util.ArrayList;

/**
 * Created by Andre on 27/05/2017.
 */
public class NeuralNetwork {

    private double momentum;
    private double learningFactor;
    private int crossValidation;
    private ArrayList<NeuronLayer> layers;

    NeuralNetwork(double momentum, double learningFactor, int numberOutputNeurons){
        this.momentum = momentum;
        this.learningFactor = learningFactor;
        layers.add(new NeuronLayer(numberOutputNeurons));

    }
}
