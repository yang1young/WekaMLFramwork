package cn.edu.zju.WekaFramework;

import weka.core.Instances;

/**
 * Created by qiaoyang on 16-6-3.
 */

public class TrainMLModels {
    public static void main(String[] args) {
        int[] nominalIndex = new int[]{1};
        DataParameters dataParameters = new DataParameters(
                "/home/qiaoyang/javaProject/WEKA_ML_Framwork/src/main/resources/log/train.arff",
                "/home/qiaoyang/javaProject/WEKA_ML_Framwork/src/main/resources/log/train.arff",
                "/home/qiaoyang/javaProject/WEKA_ML_Framwork/src/main/resources/log/",
                "/home/qiaoyang/javaProject/WEKA_ML_Framwork/src/main/resources/model/",
                90,nominalIndex,-1,0);
        DataHelper dataHelper = new DataHelper(dataParameters);
        Instances[] data = dataHelper.getDataSet();

        ModelParameters modelParameters = new ModelParameters("RF",new String[]{"-I","10"},dataHelper.getAttributes());
        RandomForestModel rf = new RandomForestModel(modelParameters,data[0]);
        //rf.crossValidate(data[0],2);
        rf.evaluateModel(data[1]);
        MLUtils.persistModel(rf,dataParameters.getModelPath(),modelParameters.getModelName());
        RandomForestModel model = (RandomForestModel) MLUtils.reloadPersistModel(dataParameters.getModelPath()+modelParameters.getModelName());
        //model.evaluateModel(data[1]);
        System.out.println(model.predictNewData(new double[]{39,23,1,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4}));
    }

}
