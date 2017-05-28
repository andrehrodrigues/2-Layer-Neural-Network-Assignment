import NeuralNetwork.NeuronLayer;
import com.mongodb.MongoClient;
import com.mongodb.client.MongoIterable;

import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        int inputDimensions = 5;//784;
        int numInputs = 10;//5000;
        int hiddenLayerSize = 25;
        int outputLayerSize = 10;
        double momentum = 0.0001;
        double learningFactor = 1.0;
        int crossValidation = 5; //Minimum value = 1 => No crossvalidation;
        double[] netError = new double[outputLayerSize];
        String gdType = "GD";
        int batchSize = 1; //If is: equal to numInput => SGD, equal to 1 => GD, equal to a 'b' value => MiniBatch GD
        DateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        Date date = new Date();

        File dataSetFile = new File("./src/data_tp1");
        InputData input = loadDataSet(dataSetFile, numInputs, inputDimensions);
        input.gaussianNormalization();

        double[] learningFactors = {0.5, 1.0, 10.0};
        int[] batchSizes = {1};//,10,50, 4000};
        int[] sizesHiddenLayer = {25, 50, 100};

        MongoClient mongo = new MongoClient( "192.168.56.101" , 27017 );

        MongoIterable<String> dbs = mongo.listDatabaseNames();
        for(String db : dbs){
            System.out.println(db);
        }

        for (int bs = 0; bs < batchSizes.length; bs++) {
            for (int lf = 0; lf < learningFactors.length; lf++) {
                for (int hl = 0; hl < sizesHiddenLayer.length; hl++) {

                    learningFactor = learningFactors[lf];
                    batchSize = batchSizes[bs];
                    hiddenLayerSize = sizesHiddenLayer[hl];
                    System.out.println("LF: "+learningFactor+"  BS: "+batchSize+"  HLS: "+hiddenLayerSize);

                    BufferedWriter bw = null;
                    FileWriter fw = null;
                    BufferedWriter vbw = null;
                    FileWriter vfw = null;

                    for (int cv = 0; cv < crossValidation; cv++) {
                        try {
                            //PrintWriter writer;
                            if (crossValidation > 1) {
                                fw = new FileWriter("./src/result/" + gdType + "in-" + numInputs + "_hLay-" + hiddenLayerSize + "_lFactor-" + learningFactor + "_batchSize-"+batchSize+"--" + sdf.format(date) + "_CV-Block-" + cv + ".csv");
                                bw = new BufferedWriter(fw);
                            } else {
                                fw = new FileWriter("./result/src/" + gdType + "in-" + numInputs + "_hLay-" + hiddenLayerSize + "_lFactor-" + learningFactor + "_batchSize-"+batchSize+"--" + sdf.format(date) + ".csv");
                                bw = new BufferedWriter(fw);
                            }

                            ArrayList<Double> lossFuncValue = new ArrayList<>();
                            lossFuncValue.add(0, 100.0);
                            int epochs = 0;

                            if (crossValidation > 1) {
                                input.setValidationData(crossValidation, cv);
                            }

                            NeuronLayer hiddenLayer = new NeuronLayer(hiddenLayerSize,
                                    inputDimensions);

                            NeuronLayer outputLayer = new NeuronLayer(outputLayerSize,
                                    hiddenLayerSize);

                            while (!(epochs > 150) && lossFuncValue.get(0) > 0.1) {

                                double[][] resultLayer1 = null;
                                double[][] resultLayerOutput = null;

                                for (int bRound = 1; bRound <= batchSize; bRound++) {
                                    if (batchSize > 1) {
                                        int begin = (input.getTrainingDataSize()/batchSize) * (bRound - 1);
                                        int end = ((input.getTrainingDataSize()/batchSize) * bRound) - 1;
                                        double[][] trainData = input.getInputForBatch(begin, end);
                                        resultLayer1 = hiddenLayer.getLayerResult(trainData);
                                    } else {
                                        resultLayer1 = hiddenLayer.getLayerResult(input.getTrainingData());
                                    }
                                    resultLayerOutput = outputLayer.getLayerResult(resultLayer1);

                                    for (int a = 0; a < input.getTrainingDataSize() / batchSize; a++) {
                                        for (int b = 0; b < outputLayerSize; b++) {
                                            netError[b] += (b == input.getExpectedTrainingOutput()[a] ? 1.0 : 0.0) - resultLayerOutput[a][b];
                                        }
                                    }

                                    for (int b = 0; b < outputLayerSize; b++) {
                                        netError[b] = netError[b] / (input.getTrainingDataSize() / batchSize);
                                    }

                                    double[] outputGradients = new double[outputLayerSize];
                                    for (int a = 0; a < input.getTrainingDataSize() / batchSize; a++) {
                                        for (int b = 0; b < outputLayerSize; b++) {
                                            outputGradients[b] += gradient(resultLayerOutput[a][b], netError[b]);
                                        }
                                    }

                                    for (int b = 0; b < outputLayerSize; b++) {
                                        outputGradients[b] = outputGradients[b] / (input.getTrainingDataSize() / batchSize);
                                    }

                                    double[] hiddenLayerGradients = new double[hiddenLayerSize + 1];
                                    for (int a = 0; a < input.getTrainingDataSize() / batchSize; a++) {
                                        for (int b = 0; b < hiddenLayerSize + 1; b++) {
                                            double sumErrors = 0.0;
                                            for (int c = 0; c < outputLayerSize; c++) {
                                                sumErrors += outputGradients[c]
                                                        * outputLayer.getWeights()[c][b];
                                            }
                                            hiddenLayerGradients[b] += gradient(resultLayer1[a][b],
                                                    sumErrors);
                                        }
                                    }

                                    for (int b = 0; b < hiddenLayerSize + 1; b++) {
                                        hiddenLayerGradients[b] = hiddenLayerGradients[b] / (input.getTrainingDataSize() / batchSize);
                                    }

                                    // Calculate the new weights for the output layer
                                    for (int a = 0; a < outputLayerSize; a++) {
                                        for (int b = 0; b < hiddenLayerSize + 1; b++) {
                                            double error = 0.0;
                                            for (int c = 0; c < (input.getTrainingDataSize() / batchSize); c++) {
                                                error += outputGradients[a] * resultLayerOutput[c][a];
                                            }
                                            error = (error / (input.getTrainingDataSize() / batchSize)) * learningFactor;// (error / numInputs) *
                                            // learningFactor;
                                            outputLayer.getWeights()[a][b] = updateWeight(
                                                    outputLayer.getWeights()[a][b],
                                                    (outputLayer.getPreviousWeights() == null ? outputLayer
                                                            .getWeights()[a][b] : outputLayer
                                                            .getPreviousWeights()[a][b]), momentum,
                                                    error);
                                        }
                                    }
                                    // Calculate the new weights for the hidden layer
                                    for (int a = 0; a < hiddenLayerSize; a++) {
                                        for (int b = 0; b < inputDimensions + 1; b++) {
                                            double error = 0.0;
                                            for (int c = 0; c < (input.getTrainingDataSize() / batchSize); c++) {
                                                error += hiddenLayerGradients[a] * resultLayer1[c][a];
                                            }
                                            error = (error / (input.getTrainingDataSize() / batchSize)) * learningFactor;// (error / numInputs) *
                                            // learningFactor;
                                            hiddenLayer.getWeights()[a][b] = updateWeight(
                                                    hiddenLayer.getWeights()[a][b],
                                                    (hiddenLayer.getPreviousWeights() == null ? hiddenLayer
                                                            .getWeights()[a][b] : hiddenLayer
                                                            .getPreviousWeights()[a][b]), momentum,
                                                    error);
                                        }
                                    }
                                }

                                //Insert the obtained loss function value for analysis.
                                //The last values will be used for checking if the network has reached a limit value.
                                lossFuncValue.add(0, lossFunction((input.getTrainingDataSize() / batchSize), outputLayerSize,
                                        input.getExpectedTrainingOutput(), getNeuralNetworkResult(input, hiddenLayer, outputLayer, false)));//resultLayerOutput));
                                if (lossFuncValue.size() > 10) {
                                    lossFuncValue.remove(10);
                                }
                                //Count epochs
                                epochs++;
                                //System.out.println("Loss Function (Erro): " + lossFuncValue.get(0) + "  epoch: " + epochs);
                                input.shuffleInput();
                                bw.write(lossFuncValue.get(0) + ";" + epochs+";");
                                for (int b = 0; b < outputLayerSize; b++) {
                                    if (b == (outputLayerSize - 1)) {
                                        bw.write(netError[b]+";");
                                    } else {
                                        bw.write(netError[b]+";");
                                    }
                                }
//                                bw.newLine();

                            //Test the net with the validation data
                            //double[][] validationAlResults = runNetwork(input, hiddenLayer, outputLayer, true);
                            //double[] validationAvgResults = runNetwork(input, hiddenLayer, outputLayer, true, false);
                            double validationLossFunctionValue = lossFunction(input.getValidationDataSize(), outputLayer.getSize(), input.getExpectedValidationOutput(),
                                    getNeuralNetworkResult(input, hiddenLayer, outputLayer, true));
//                            vfw = new FileWriter("./src/result/validation/validation-" + gdType + "-in-" + numInputs + "_hLay-" + hiddenLayerSize + "_lFactor-" + learningFactor +"_cv-"+cv+".csv",true);
//                            vbw = new BufferedWriter(fw);
                            bw.write(validationLossFunctionValue + ";");
                            bw.newLine();

//                            if (vbw != null)
//                                vbw.close();
//
//                            if (vfw != null)
//                                vfw.close();

                            }//End of while that repeats the network calculating

                            //Close and save the file data.
                            if (bw != null)
                                bw.close();

                            if (fw != null)
                                fw.close();

                        } catch (IOException e) {
                            System.out.println(e);
                        }//End of the TRY-CATCH block with the execution code.

                    }//End of the cross validation loop.

        //All process is done inside 3 fors to change te parameters and rerun the net.
                }
            }
        }

    }

    public static InputData loadDataSet(File file, int n, int m) {

        InputData inData = new InputData(n, n, m + 1);

        double sumValues = 0.0;

        Scanner sc = null;
        try {
            sc = new Scanner(file);

            for (int a = 0; a < n; a++) {
                if (sc.hasNext()) {
                    Scanner lineSC = new Scanner(sc.nextLine())
                            .useDelimiter(",");
                    for (int b = 0; b < m + 1; b++) {
                        if (lineSC.hasNextDouble()) {
                            if (b == 0) {
                                inData.expectedOutput[a] = lineSC.nextDouble();
                                inData.fullDataSet[a][m] = 1.0;
                                inData.trainingData[a][m] = 1.0;
                            } else {
                                inData.fullDataSet[a][b - 1] = lineSC.nextDouble();
                                inData.trainingData[a][b - 1] = lineSC.nextDouble();
                                sumValues += inData.trainingData[a][b - 1];
                            }
                        }
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            sc.close();
        }

        inData.setMean(sumValues / (n * m));

        return inData;
    }

    public static double lossFunction(int numEntradas, int numSaidas, double[] expectedOutput, double[][] networkOutput) {
        double result = 0.0;
        for (int a = 0; a < numEntradas; a++) {
            for (int b = 0; b < numSaidas; b++) {
                result += ((-1 * ((expectedOutput[a] == new Double(numSaidas).doubleValue() + 1.0 ? expectedOutput[a] : 0)))
                        * (Math.log(networkOutput[a][b])) - (1 - (expectedOutput[a] == new Double(numSaidas).doubleValue() + 1.0 ? expectedOutput[a] : 0))
                        * (Math.log((1 - networkOutput[a][b]))));
            }
        }
        result = result / numEntradas;

        return result;
    }

    public static double gradient(double output, double netError) {
        return (output * (1 - output) * netError);
    }

    public static double updateWeight(double oldWeight, double olderWeight, double momentum, double error) {
        return oldWeight + (momentum * olderWeight) + error;
    }

    public static double[][] getNeuralNetworkResult(InputData input, NeuronLayer hiddenLayer, NeuronLayer outputLayer, boolean validation) {

        double[][] resultLayer1;
        if (validation) {
            resultLayer1 = hiddenLayer.getLayerResult(input.getValidationData());
        } else {
            resultLayer1 = hiddenLayer.getLayerResult(input.getTrainingData());
        }

        return outputLayer.getLayerResult(resultLayer1);
    }

    public static double[][] runNetwork(InputData input, NeuronLayer hiddenLayer, NeuronLayer outputLayer, boolean validation) {

        int inputSize;
        if (validation) {
            inputSize = input.getValidationDataSize();
        } else {
            inputSize = input.getTrainingDataSize();
        }

        double[][] resultLayerOutput = getNeuralNetworkResult(input, hiddenLayer, outputLayer, validation);

        double[][] netError = new double[inputSize][outputLayer.getSize()];

        for (int a = 0; a < inputSize; a++) {
            for (int b = 0; b < outputLayer.getSize(); b++) {
                netError[a][b] = (b == input.getExpectedOutput()[a] ? 1.0 : 0.0) - resultLayerOutput[a][b];
            }
        }

        return netError;
    }

    public static double[] runNetwork(InputData input, NeuronLayer hiddenLayer, NeuronLayer outputLayer, boolean validation, boolean takeMean) {

        int inputSize;
        if (validation) {
            inputSize = input.getValidationDataSize();
        } else {
            inputSize = input.getTrainingDataSize();
        }

        double[][] resultLayerOutput = getNeuralNetworkResult(input, hiddenLayer, outputLayer, validation);

        double[] netError = new double[outputLayer.getSize()];

        for (int a = 0; a < inputSize; a++) {
            for (int b = 0; b < outputLayer.getSize(); b++) {
                netError[b] += (b == input.getExpectedOutput()[a] ? 1.0 : 0.0) - resultLayerOutput[a][b];
            }
        }

        if (takeMean) {
            for (int b = 0; b < outputLayer.getSize(); b++) {
                netError[b] = netError[b] / input.getTrainingDataSize();
            }
        }

        return netError;
    }

}
