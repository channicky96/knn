package knn;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import static java.util.stream.Collectors.toMap;
import java.util.stream.Stream;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class KNN extends AbstractClassifier {

    // K value(s) to used for classifying
    private int k, maxk, ensembleSize;
    Instances train, test;
    boolean toStandardise, useLOOCV, useWeighted;

    /**
     * constructor for the knn classifier
     */
    public KNN() {
        this.k = 1; // defaults to use 1 k i.e. nearest neighbour
        this.maxk = 100;
        this.train = null;
        this.test = null;
        this.toStandardise = true; // defaults to stadardise attributes
        this.useLOOCV = false;
        this.useWeighted = false; // defaults to not use weighted voting scheme
        this.ensembleSize = 50; // create knn ensemble with this default size
    }

    public static void main(String[] args) throws Exception {

        String path = "benchmarks";

//        String[] datasets = {"bank", "blood", "breast-cancer-wisc-diag", "breast-tissue",
//            "cardiotocography-10clases", "conn-bench-sonar-mines-rocks", "conn-bench-vowel-deterding",
//            "ecoli", "glass", "hill-valley", "image-segmentation", "ionosphere", "iris", "libras",
//            "oocytes_merluccius_nucleus_4d", "oocytes_trisopterus_states_5b", "optical", "ozone",
//            "page-blocks", "parkinsons", "planning", "post-operative", "ringnorm", "seeds", "spambase",
//            "statlog-landsat", "statlog-vehicle", "steel-plates", "synthetic-control", "twonorm",
//            "vertebral-column-3clases", "wall-following", "waveform-noise", "wine-quality-white", "yeast"};
        String[] datasets = {"nepenthes"};
//        String[] datasets = {"conn-bench-vowel-deterding"};

        for (String dataset : datasets) {
            System.out.println("Processing dataset: " + dataset);
            Instances train = loadData(path + "/" + dataset + "/" + dataset + "_TRAIN");
            Instances test = loadData(path + "/" + dataset + "/" + dataset + "_TEST");

            double correct = 0.0, acc, totalAcc = 0.0;
            KNN knn = new KNN();
            knn.setK(3);
            knn.setStandardise(true);
            knn.setWeighted(false);
            knn.setLOOCV(false);
            knn.buildClassifier(train, test);

            ///////////////////////////////  test harness for Q2 /////////////////////////////////
            for (int i = 0; i < test.numInstances(); i++) {
                if (knn.classifyInstance(knn.test.get(i)) == knn.test.get(i).classValue()) {
                    correct++;
                }
            }
            acc = (correct / knn.test.numInstances()) * 100;
            System.out.println("Accurarcy: " + acc);
            System.out.println("-------------------------");
            
/////////////////////////////////// ensemble ////////////////////////////

//            KnnEnsemble KnnEnsemble = new KnnEnsemble(knn);
//            ArrayList<KNN> ensemble = KnnEnsemble.buildKnnEnsemble();
//            for (KNN knnClassifier : ensemble) {
//                for (int i = 0; i < knnClassifier.test.numInstances(); i++) {
//                    if (knnClassifier.classifyInstance(knnClassifier.test.get(i)) == knnClassifier.test.get(i).classValue()) {
//                        correct++;
//                    }
//                }
//                totalAcc += (correct / knnClassifier.test.numInstances());
//                correct = 0.0;
//            }
//            System.out.println(totalAcc / ensemble.size());
//            System.out.println("\nEnsemble size: " + ensemble.size());
//            System.out.println("Ensemble accurarcy: " + totalAcc / ensemble.size());
//            System.out.println("========================================================");
            
            /////////////////////////////Q3//////////////////////////////
            
//            NaiveBayes nb = new NaiveBayes();               
//            RandomForest randFor = new RandomForest();       
//            IBk ibk = new IBk(1);
//            J48 c45 = new J48();
//            nb.buildClassifier(train);
//            for (int i = 0; i < test.numInstances(); i++) {
//                if (nb.classifyInstance(test.get(i)) == test.get(i).classValue()) {
//                    correct++;
//                }
//            }
//            acc = (correct / test.numInstances());
//            System.out.println(1-acc);
        }
    }

    /**
     *
     * @param train to build the classifier and to standardise
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances train) throws Exception {

        // Standardising attributes
        if (this.toStandardise) {
            Filter standardise = new Standardize();
            standardise.setInputFormat(train);
            this.train = Filter.useFilter(train, standardise);
        } else {
            this.train = train;
        }

        // Setting max k
        if (train.numInstances() * 0.2 < 100) {
            this.maxk = (int) (train.numInstances() * 0.2);
        }

        // Determine optimal k for the training set through LOOCV
        if (useLOOCV) {
            this.setK(LOOCV());
        }
    }

    /**
     *
     * @param train to build the classifier and to standardise
     * @param test to standardise using train data's mean and sd
     * @throws Exception
     */
    public void buildClassifier(Instances train, Instances test) throws Exception {

        // Standardising attributes
        if (this.toStandardise) {
            Filter standardise = new Standardize();
            standardise.setInputFormat(train);
            this.train = Filter.useFilter(train, standardise);
            this.test = Filter.useFilter(test, standardise);
        } else {
            this.train = train;
            this.test = test;
        }

        // Setting max k
        if (train.numInstances() * 0.2 < 100) {
            this.maxk = (int) (train.numInstances() * 0.2);
        }

        // Determine optimal k for the training set through LOOCV 
        if (useLOOCV) {
            this.setK(LOOCV());
        }
    }

    @Override
    public double classifyInstance(Instance queryInstance) throws Exception {

        HashMap map = returnMap(queryInstance);
        ArrayList<Integer> knnArray = new ArrayList<>();

        if (!this.useWeighted) {
            // Insert k amount of least distance cases' class value
            Object[] tempArray = map.entrySet().stream().sorted(Map.Entry.comparingByValue()).toArray();

            for (int j = 0; j < this.k; j++) {
                knnArray.add((int) (this.train.get(Integer.valueOf(tempArray[j].toString().split("=")[0])).classValue()));
            }
        } else {
            Map<Double, Integer> sortedMap = new TreeMap<>(map);

            // Obtain all unique class values
            double[] temp = train.attributeToDoubleArray(train.classIndex());
            Set<Double> mySet = new HashSet<>();
            for (double classVal : temp) {
                mySet.add(classVal);
            }

            double[] classVec = new double[mySet.size()];
            int k = 0;
            for (Double classVal : mySet) {
                classVec[k++] = classVal;
            }

            double[] classVotes = new double[train.classAttribute().numValues()];

            // add up weighed vote for each class
            Object[] tempArray = sortedMap.entrySet().toArray();
            for (int i = 0; i < classVec.length; i++) {
                for (int j = tempArray.length - 1; j > tempArray.length - 1 - this.k; j--) {
                    if (Double.valueOf(tempArray[j].toString().split("=")[1]) == classVec[i]) {
                        classVotes[i] += Double.valueOf(tempArray[j].toString().split("=")[0]);
                    }
                }
            }

            // return predicted class as class with highest weight score
            double max = Arrays.stream(classVotes).max().getAsDouble();
            for (int i = 0; i < classVotes.length; i++) {
                if (classVotes[i] == max) {
                    return classVec[i];
                }
            }

        }
        return findMode(knnArray);
    }

    /**
     *
     * @param queryInstance the test instance for computing the distance
     * @return a hash map with k(index of the train instance) as key and
     * distance or weight as value
     * @throws Exception
     */
    public HashMap returnMap(Instance queryInstance) throws Exception {
        HashMap map = new HashMap<>();
        for (int j = 0; j < this.train.numInstances(); j++) { // for each instance 
            double dist = 0.0;
            Instance testInstance = this.train.get(j);
            double[] queryArray = queryInstance.toDoubleArray();
            double[] testArray = testInstance.toDoubleArray();

            // Compute the Euclidean distance between the query case and test case
            for (int i = 0; i < queryArray.length - 1; i++) {
                dist += Math.pow(queryArray[i] - testArray[i], 2);
            }

            // Use k as key and distance or weight as value
            if (!this.useWeighted) {
                map.put(j, dist);
            } else {
                map.put((1 / (1 + dist)), testInstance.classValue());
            }
        }
        return map;
    }

    /**
     *
     * @return the "optimal" k value found
     * @throws Exception
     */
    public int LOOCV() throws Exception {

        TreeMap<Integer, Double> map = new TreeMap<>();
        // For each k, compute error rates
        for (int j = 1; j < this.maxk + 1; j++) {
            this.setK(j);
            int correct = 0;
            // for each instance in training data, partition data and classify
            for (int i = 0; i < this.train.numInstances(); i++) {
                Instance thisInstance = this.train.get(i);
                this.train.remove(i);
                if (classifyInstance(thisInstance) == thisInstance.classValue()) {
                    correct++;
                }
                // insert the data back in
                this.train.add(i, thisInstance);
            }
            map.put(j, 1 - (double) correct / (this.train.numInstances() - 1));
        }

        // Return the k with the minimum error rate
        Object[] tempKeyArray = map.keySet().toArray();
        Object[] tempValArray = map.values().toArray();
        ArrayList<Integer> results = new ArrayList<>();

        for (int i = 0; i < tempValArray.length - 1; i++) {
            if (tempValArray[0] == tempValArray[i]) {
                results.add((int) tempKeyArray[i]);
            }
        }

        if (results.size() == 1) {
            return results.get(0);
        } else { // ties are settled randomly
            Random ran = new Random();
            return results.get(ran.nextInt(results.size()));
        }
    }

    /**
     *
     * @param queryInstance
     * @return the proportion of the neighbours voting for each response
     * variable value.
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance queryInstance) throws Exception {

        ArrayList<Double> results = new ArrayList<>();
        double predicted = this.classifyInstance(queryInstance);
        double[] classVec = new double[train.classAttribute().numValues()];

        // Obtain all unique class values
        for (int j = 0; j < train.classAttribute().numValues(); j++) {
            classVec[j] = train.get(j).classValue();
        }
        // put queryInstance's predicted class value as the first element
        // as this will be used to calculate  the first element to be displayed
        for (int i = 0; i < classVec.length; i++) {
            if (classVec[i] == predicted) {
                double tmp = classVec[0];
                classVec[i] = tmp;
                classVec[0] = predicted;
                break;
            }
        }

        // Classify all instances to get a distrubution
        for (int i = 0; i < train.numInstances(); i++) {
            results.add(this.classifyInstance(this.train.get(i)));
        }

        // Compute proportion for each response variable
        double[] distribution = new double[classVec.length];
        for (int i = 0; i < classVec.length; i++) {
            distribution[i] = (double) (Collections.frequency(results, classVec[i])) / results.size();
        }

        return distribution;
    }

    /**
     *
     * @param knnArray an array of k nearest predicted classes
     * @return mode of knnArray, i.e. the predicted class value
     */
    public double findMode(ArrayList<Integer> knnArray) {
        HashMap<Integer, Integer> hmap = new HashMap<>();
        ArrayList<Integer> modeSet = new ArrayList<>();
        int maxOccurrences = 0, mode = 0;
        for (int num : knnArray) {
            hmap.merge(num, 1, Integer::sum);
            int occurrences = hmap.get(num);
            if (occurrences > maxOccurrences || (occurrences == maxOccurrences && num < mode)) {
                maxOccurrences = occurrences;
                mode = num;
                modeSet.clear();
                modeSet.add(num);
            } else if (occurrences == maxOccurrences) {
                modeSet.add(num);
            }
        }

        if (Collections.frequency(hmap.values(), maxOccurrences) == 1) {
            return (double) mode;
        } else {
            Random r = new Random();
            return (double) (modeSet.get(r.nextInt(modeSet.size())));
        }
    }

    /**
     *
     * @return max k allowed
     */
    public int getMaxk() {
        return maxk;
    }

    /**
     *
     * @param maxk maximum k allowed
     */
    public void setMaxk(int maxk) {
        this.maxk = maxk;
    }

    /**
     *
     * @return if the standardise data option is enabled
     */
    public boolean isToStandardise() {
        return toStandardise;
    }

    /**
     *
     * @param toStandardise to set the standardise data option
     */
    public void setToStandardise(boolean toStandardise) {
        this.toStandardise = toStandardise;
    }

    /**
     *
     * @return if the LOOCV to set k option is enabled
     */
    public boolean isUseLOOCV() {
        return useLOOCV;
    }

    /**
     *
     * @param useLOOCV to set the LOOCV to set k option
     */
    public void setUseLOOCV(boolean useLOOCV) {
        this.useLOOCV = useLOOCV;
    }

    /**
     *
     * @return if the use weight voting scheme to set k option is enabled
     */
    public boolean isUseWeighted() {
        return useWeighted;
    }

    /**
     *
     * @param useWeighted to set the use weighting scheme to set k option
     */
    public void setUseWeighted(boolean useWeighted) {
        this.useWeighted = useWeighted;
    }

    /**
     *
     * @param kToUse the amount of k to use
     */
    public void setK(int kToUse) {
        this.k = kToUse;
    }

    /**
     *
     * @param train the train instances
     */
    public void setTrain(Instances train) {
        this.train = train;
    }

    /**
     *
     * @param test the train instances
     */
    public void setTest(Instances test) {
        this.test = test;
    }

    /**
     *
     * @param toStandardise to standardise the attributes or not
     */
    public void setStandardise(boolean toStandardise) {
        this.toStandardise = toStandardise;
    }

    /**
     *
     * @param useLOOCV to set k through LOOCV or not
     */
    public void setLOOCV(boolean useLOOCV) {
        this.useLOOCV = useLOOCV;
    }

    /**
     *
     * @param useWeighted to use weighted voting scheme or not
     */
    public void setWeighted(boolean useWeighted) {
        this.useWeighted = useWeighted;
    }

    /**
     *
     * @param size to set the knn ensemble size
     */
    public void setEnsembleSize(int size) {
        this.ensembleSize = size;
    }

    /**
     *
     * @param fullPath
     * @return object of type Instances of the data loaded
     * @throws Exception
     */
    public static Instances loadData(String fullPath) throws Exception {
        Instances output = new Instances(new FileReader(fullPath + ".arff"));
        output.setClassIndex(output.numAttributes() - 1);
        return output;
    }

}
