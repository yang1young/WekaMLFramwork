package cn.edu.zju.WekaFramework;

import java.io.Serializable;
import java.util.ArrayList;

import weka.core.*;
/**
 * Created by qiaoyang on 17-6-3.
 */
public interface Parameters {
    public abstract String paraToString();
}

class DataParameters implements Parameters,Serializable {
    private String trainDataPath;
    private String testDataPath;
    private String logPath;
    private String modelPath;
    private double samplePercent;
    private int[] nominalIndex;
    private int labelIndex;
    private int sampleLabel;

    DataParameters(String trainDataPath, String testDataPath, String logPath, String modelPath, double samplePercent,int[] nominalIndex,int labelIndex,int sampleLabel) {
        this.trainDataPath = trainDataPath;
        this.testDataPath = testDataPath;
        this.logPath = logPath;
        this.modelPath = modelPath;
        this.samplePercent = samplePercent;
        this.nominalIndex = nominalIndex;
        this.labelIndex = labelIndex;
        this.sampleLabel = sampleLabel;
    }

    public int getLabelIndex() {
        return labelIndex;
    }

    public void setLabelIndex(int labelIndex) {
        this.labelIndex = labelIndex;
    }

    public String paraToString() {
        return Double.toString(samplePercent);
    }

    public String getTrainDataPath() {
        return trainDataPath;
    }

    public void setTrainDataPath(String trainDataPath) {
        this.trainDataPath = trainDataPath;
    }

    public int getSampleLabel() {
        return sampleLabel;
    }

    public void setSampleLabel(int sampleLabel) {
        this.sampleLabel = sampleLabel;
    }

    public int[] getNominalIndex() {
        return nominalIndex;
    }

    public void setNominalIndex(int[] nominalIndex) {
        this.nominalIndex = nominalIndex;
    }

    public String getTestDataPath() {
        return testDataPath;
    }

    public void setTestDataPath(String testDataPath) {
        this.testDataPath = testDataPath;
    }

    public String getLogPath() {
        return logPath;
    }

    public void setLogPath(String logPath) {
        this.logPath = logPath;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public double getSamplePercent() {
        return samplePercent;
    }

    public void setSamplePercent(double samplePercent) {
        this.samplePercent = samplePercent;
    }
}

class ModelParameters implements Parameters,Serializable {
    private String modelName;
    private String[] modelParam;
    private ArrayList<Attribute> attributes;


    ModelParameters(String modelName, String[] modelParam,ArrayList<Attribute> attributes) {
        this.modelName = modelName;
        this.modelParam = modelParam;
        this.attributes = attributes;
    }

    public ArrayList<Attribute> getAttributes() {
        return attributes;
    }

    public void setAttributes(ArrayList<Attribute> attributes) {
        this.attributes = attributes;
    }


    public String getModelName() {
        return modelName;
    }

    public String[] getModelParam() {
        return modelParam;
    }

    public String paraToString() {
        return modelName;
    }
}
