package cn.edu.zju.WekaFramework;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.ArrayList;

/**
 * Created by qiaoyang on 16-6-3.
 */
public class DataHelper {
    private DataParameters dataParameters;
    private ArrayList<Attribute> attributes;
    DataHelper(DataParameters dataParameters) {
        this.dataParameters = dataParameters;
    }

    public Instances[] getDataSet() {
        Instances[] result = new Instances[2];
        try {
            String trainPath = dataParameters.getTrainDataPath();
            Instances trainData = new ConverterUtils.DataSource(trainPath).getDataSet();
            String testPath = dataParameters.getTestDataPath();
            Instances testData = new ConverterUtils.DataSource(testPath).getDataSet();

            if(dataParameters.getLabelIndex()==-1) {
                trainData.setClassIndex(trainData.numAttributes() - 1);
                testData.setClassIndex(trainData.numAttributes() - 1);
            }
            else {
                trainData.setClassIndex(dataParameters.getLabelIndex());
                testData.setClassIndex(dataParameters.getLabelIndex());

            }

            if (!trainPath.contains(".arff")) {
                NumericToNominal convert = new NumericToNominal();
                convert.setAttributeIndicesArray(dataParameters.getNominalIndex());
                convert.setInputFormat(trainData);
                trainData = Filter.useFilter(trainData, convert);
                testData = Filter.useFilter(testData, convert);
            }
            if (dataParameters.getSamplePercent() != 0) {
                result[0] = MLUtils.sample(trainData,dataParameters.getSamplePercent(),dataParameters.getSampleLabel());
            }
            else{
                result[0] = trainData;
            }
            result[1] = testData;
            ArrayList<Attribute> atts = new ArrayList<Attribute>();
            for (int i = 0; i < trainData.numAttributes(); i++) {
                atts.add(trainData.attribute(i));
            }
            setAttributes(atts);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    public ArrayList<Attribute> getAttributes() {
        return attributes;
    }

    private void setAttributes(ArrayList<Attribute> attributes) {
        this.attributes = attributes;
    }
/*  Standardize standardize = new Standardize();
            standardize.setInputFormat(data);
            Instances tempData = Filter.useFilter(data,standardize);

            Normalize normalize = new Normalize();
            String[] normalOptions = new String[]{"-S","2.0","-T","-1.0"};
            normalize.setOptions(normalOptions);
            normalize.setInputFormat(tempData);
            newData = Filter.useFilter(tempData,normalize);
            */
}