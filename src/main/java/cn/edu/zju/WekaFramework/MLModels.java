package cn.edu.zju.WekaFramework;

import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.LibSVM;
import weka.core.*;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;


/**
 * Created by yangqiao on 15/9/24.
 */
public class MLModels implements Serializable {
    protected static Logger logger = Logger.getLogger(AbstractClassifier.class);
    protected AbstractClassifier model;
    protected ModelParameters modelParameters;
    protected ArrayList<Attribute> attribute;

    MLModels(ModelParameters modelParameters) {
        this.modelParameters = modelParameters;
    }

    public double predictNewData(double[] data) {

            Instances df = new Instances("predictData", modelParameters.getAttributes(), 0);
            df.setClassIndex(df.numAttributes() - 1);
            Instance sample = new DenseInstance(1.0, data);
            sample.setDataset(df);
            return predict(sample);
    }

    public double predict(Instance sample) {
        double res = -1;
        try {
            res = model.classifyInstance(sample);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return res;
    }

    public double[] predictProb(Instance sample) {
        double[] res = new double[2];
        res[0] = -1;
        res[1] = -1;
        try {
            res = model.distributionForInstance(sample);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return res;
    }

    public void crossValidate(Instances trainSet, int fold) {
        try {
            Evaluation eval = new Evaluation(trainSet);
            eval.crossValidateModel(model, trainSet, fold, new Random(0));
            logger.info("summary:" + eval.toSummaryString("\nResults\n\n", false));
            logger.info("details:\n " + eval.toClassDetailsString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluateModel(Instances testData){
        try {
            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(model,testData);
            logger.info("summary:" + eval.toSummaryString("\nResults\n\n", false));
            logger.info("details:\n " + eval.toClassDetailsString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Attribute> getAttribute() {
        return attribute;
    }

    public void setAttribute(ArrayList<Attribute> attribute) {
        this.attribute = attribute;
    }
}


class RandomForestModel extends MLModels {

    public RandomForestModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new RandomForest();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


class NueralNetworkModel extends MLModels {

    public NueralNetworkModel(ModelParameters modelParameters,  Instances data) {
        super(modelParameters);
        model = new MultilayerPerceptron();
        try {
            model.buildClassifier(data);
            model.setOptions(modelParameters.getModelParam());
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}


class SVMModel extends MLModels {
    public SVMModel(ModelParameters modelParameters,  Instances data) {
        super(modelParameters);
        model = new LibSVM();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


class CARTModel extends MLModels {

    public CARTModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new J48();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

class SMOModel extends MLModels {

    public SMOModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new SMO();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

class LogisticModel extends MLModels {

    public LogisticModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new Logistic();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class NaiveBayesModel extends MLModels {

    public NaiveBayesModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new NaiveBayes();
        try {
            model.setOptions(modelParameters.getModelParam());
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class AdaBoostModel extends MLModels {

    public AdaBoostModel(ModelParameters modelParameters, Instances data) {
        super(modelParameters);
        model = new AdaBoostM1();
        try {
            model.buildClassifier(data);
            model.setOptions(modelParameters.getModelParam());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}



