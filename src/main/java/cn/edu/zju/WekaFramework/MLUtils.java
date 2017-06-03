package cn.edu.zju.WekaFramework;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.AbstractFileSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

import java.io.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by qiaoyang on 17-6-3.
 */
public class MLUtils {

    public static void fileTypeConverter(String inputFile, String outputFile) {
        File input = new File(inputFile);
        File output = new File(outputFile);
        AbstractFileLoader loader = ConverterUtils.getLoaderForFile(input);
        AbstractFileSaver saver = ConverterUtils.getSaverForFile(output);
        try {
            loader.setSource(input);
            Instances data = loader.getDataSet();
            System.out.println(data);
            saver.setInstances(data);
            saver.setFile(output);
            saver.setDestination(output);
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void persistModel(MLModels model, String modelPath, String modelName) {
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new FileOutputStream(modelPath + modelName));
            oos.writeObject(model);
            oos.flush();
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MLModels reloadPersistModel(String ModelPath) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(new File(ModelPath)));
            MLModels model = (MLModels) ois.readObject();
            return model;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }


    public static Instances sample(Instances data, double percent, int sampleLabel) {
        String[] sampleOptions = {"-Z", Double.toString(percent)};

        Instances classP = new Instances(data, 0);
        Instances classN = new Instances(data, 0);
        long totalNum = data.numInstances();

        for (int i = 0; i < totalNum; i++) {
            Instance temp = data.instance(i);
            if (temp.classValue() == sampleLabel) {
                classP.add(temp);
            } else {
                classN.add(temp);
            }
        }
        classP = boostrapSample(classP, sampleOptions);
        classP.addAll(classN);
        return classP;
    }


    private static Instances boostrapSample(Instances data, String[] options) {
        Resample convert = new Resample();
        try {
            convert.setOptions(options);
            convert.setInputFormat(data);
            data = Filter.useFilter(data, convert);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return data;
    }

    public static void libsvmToCsv(String inputFile, String outputFile, int numOfAttributes, ArrayList indexOfNominal) {
        BufferedReader br = null;
        BufferedWriter bw = null;
        String s;
        Pattern patternIsNum = Pattern.compile("[0-9.]*");
        Pattern patternIsSciNotation = Pattern.compile("^((\\d+.?\\d+)[E]{1}(\\d+))$");
        long lineNum = 0;

        try {
            br = new BufferedReader(new FileReader(inputFile));
            bw = new BufferedWriter(new FileWriter(outputFile));
            int errorCount = 0;
            do {
                s = br.readLine();
                lineNum++;
                boolean errorFlag = false;
                if (s == null)
                    break;
                String[] result = new String[numOfAttributes + 1];
                StringBuffer sb = new StringBuffer();
                for (int i = 0; i < numOfAttributes; i++) {
                    result[i] = "0,";
                }

                String temp[] = s.split(" ");
                result[numOfAttributes] = temp[0];
                for (int i = 1; i < temp.length; i++) {
                    String[] innerTemp = temp[i].split(":");
                    char[] indexArray = innerTemp[0].toCharArray();
                    if (innerTemp.length > 1) {
                        int attributesIndex = Integer.parseInt(innerTemp[0]);
                        Matcher isSciNotation = patternIsSciNotation.matcher(innerTemp[1]);
                        Matcher isNum = patternIsNum.matcher(innerTemp[1]);

                        if (isSciNotation.matches()) {
                            result[attributesIndex - 1] = new BigDecimal(innerTemp[1]).toPlainString() + ",";
                        } else if (isNum.matches() || indexOfNominal.contains(attributesIndex)) {
                            result[attributesIndex - 1] = innerTemp[1] + ",";
                        } else {
                            errorFlag = true;
                        }
                    }
                }
                if (errorFlag) {
                    errorCount++;
                    System.out.println("第" + errorCount + "条-----第 " + lineNum + "行------ " + s);
                }
                for (int j = 0; j < numOfAttributes + 1; j++) {
                    sb.append(result[j]);
                }
                bw.write(sb.toString());
                bw.newLine();
            } while (s != null);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                br.close();
                bw.flush();
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }
}
