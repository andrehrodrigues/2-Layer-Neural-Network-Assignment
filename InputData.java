import java.util.Random;

/**
 * Created by Andre on 13/05/2017.
 */
public class InputData {

    int numInputs;
    int inputDimensions;

    double[] expectedOutput;

    double[][] fullDataSet;

    double[][] trainingData;
    double[] expectedTrainingOutput;
    int trainingDataSize;

    double[][] validationData;
    double[] expectedValidationOutput;
    int validationDataSize;

    double mean;
    double stdDev;

    InputData(int expOutSize, int numInputs, int inputDimensions){
        this.inputDimensions = inputDimensions;
        this.numInputs = numInputs;
        this.trainingDataSize = numInputs;
        this.validationDataSize = 0;
        this.expectedOutput = new double[expOutSize];
        this.fullDataSet = new double[numInputs][inputDimensions];
        this.trainingData = new double[numInputs][inputDimensions];
    }

    public void shuffleInput(){
        Random rnd = new Random();
        for (int i = this.trainingData.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            double[] a = this.trainingData[index];
            this.trainingData[index] = a;
            this.trainingData[i] = a;
        }
    }
    
    public void gaussianNormalization(){
    	double stdDev = 0;
    	for (int a = 0; a < this.fullDataSet.length; a++) {
            for (int b = 0; b < this.fullDataSet[0].length; b++) {
                stdDev += Math.pow(this.fullDataSet[a][b] - this.mean, 2);
            }
        }
    	
    	stdDev = stdDev/ (this.fullDataSet.length* this.fullDataSet[0].length);
    	stdDev = Math.sqrt(stdDev);

    	this.stdDev = stdDev;
    	
    	for (int a = 0; a < this.fullDataSet.length; a++) {
            for (int b = 0; b < this.fullDataSet[0].length; b++) {
                this.fullDataSet[a][b] = (this.fullDataSet[a][b] - mean)/stdDev;
            }
        }
    }

    public void setValidationData(int numFolds, int foldBlockNumber){

        this.validationDataSize = numInputs/numFolds;
        this.trainingDataSize = numInputs - this.validationDataSize;

        int foldEnd = (this.validationDataSize*(1+foldBlockNumber)) -1 ;
        int foldBegin = (foldEnd + 1) - this.validationDataSize ;

        this.validationData = new double[validationDataSize][inputDimensions];
        this.expectedValidationOutput = new double[validationDataSize];
        this.expectedTrainingOutput = new double[this.trainingDataSize];
        double[][] newTrainingSet = new double[this.trainingDataSize][this.inputDimensions];

        for (int a = 0; a < numInputs; a++) {
            for (int b = 0; b < this.inputDimensions; b++) {
                if(a >= foldBegin && a <= foldEnd){
                    this.validationData[ a - ((this.validationDataSize)*foldBlockNumber) ][b] = this.fullDataSet[a][b];
                }else if(a < foldBegin){
                    newTrainingSet[a][b] = this.fullDataSet[a][b];
                }else{
                    newTrainingSet[a- this.validationDataSize][b] = this.fullDataSet[a][b];
                }
            }
        }

        for (int a = 0; a < numInputs; a++) {
            if(a >= foldBegin && a <= foldEnd){
                this.expectedValidationOutput[ a - (this.validationDataSize)*foldBlockNumber ] = this.expectedOutput[a];
            }else if(a < foldBegin){
                this.expectedTrainingOutput[a] = this.expectedOutput[a];
            }else{
                this.expectedTrainingOutput[a- this.validationDataSize] = this.expectedOutput[a];
            }
        }

        this.trainingData = newTrainingSet;
    }

    public double[][] getInputForBatch(int begin, int end){
        double[][] batchData = new double[end - begin + 1][this.getInputDimensions()];
        for (int a = begin; a < end; a++) {
            for (int b = 0; b < this.inputDimensions; b++) {
                batchData[a-begin][b] = this.trainingData[a][b];
            }
        }
        return batchData;
    }

    public double[][] getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(double[][] trainingData) {
        this.trainingData = trainingData;
    }

    public double[] getExpectedOutput() {
        return expectedOutput;
    }

    public void setExpectedOutput(double[] expectedOutput) {
        this.expectedOutput = expectedOutput;
    }

	public double getMean() {
		return mean;
	}

	public void setMean(double mean) {
		this.mean = mean;
	}

    public int getNumInputs() {
        return numInputs;
    }

    public void setNumInputs(int numInputs) {
        this.numInputs = numInputs;
    }

    public int getInputDimensions() {
        return inputDimensions;
    }

    public void setInputDimensions(int inputDimensions) {
        this.inputDimensions = inputDimensions;
    }

    public double[][] getFullDataSet() {
        return fullDataSet;
    }

    public void setFullDataSet(double[][] fullDataSet) {
        this.fullDataSet = fullDataSet;
    }

    public int getTrainingDataSize() {
        return trainingDataSize;
    }

    public void setTrainingDataSize(int trainingDataSize) {
        this.trainingDataSize = trainingDataSize;
    }

    public double[][] getValidationData() {
        return validationData;
    }

    public void setValidationData(double[][] validationData) {
        this.validationData = validationData;
    }

    public int getValidationDataSize() {
        return validationDataSize;
    }

    public void setValidationDataSize(int validationDataSize) {
        this.validationDataSize = validationDataSize;
    }

    public double getStdDev() {
        return stdDev;
    }

    public void setStdDev(double stdDev) {
        this.stdDev = stdDev;
    }

    public double[] getExpectedTrainingOutput() {
        return expectedTrainingOutput;
    }

    public void setExpectedTrainingOutput(double[] expectedTrainingOutput) {
        this.expectedTrainingOutput = expectedTrainingOutput;
    }

    public double[] getExpectedValidationOutput() {
        return expectedValidationOutput;
    }

    public void setExpectedValidationOutput(double[] expectedValidationOutput) {
        this.expectedValidationOutput = expectedValidationOutput;
    }
}
