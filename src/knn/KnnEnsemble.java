package knn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;
import static java.util.stream.Collectors.toMap;
import weka.core.Debug.Random;
import weka.core.Instances;

public class KnnEnsemble {

    // variables to represent a kNN ensemble
    int ensembleSize;
    ArrayList<KNN> ensemble;
    Instances train, test;

    // empty constructor
    public KnnEnsemble() {

    }

    // construct for a knn ensemble, default size to 50
    public KnnEnsemble(KNN knn) {
        this.train = knn.train;
        this.test = knn.test;
        this.ensembleSize = 50;
        this.ensemble = new ArrayList<>();
    }

    /**
     *
     * @return an ensemble of the knn classifier
     */
    public ArrayList<KNN> buildKnnEnsemble() throws Exception {

        int ranInt;
        Random random = new Random();
        int numInts = this.train.numInstances();
        int numOfEnsemInstance = (int) (this.train.numInstances() * 0.632);

        // Ensemble of KNN classifiers
        ArrayList<KNN> KnnEnsemble = new ArrayList<>();

        // Unique bagged instances indexes
        SortedSet<Integer> intSet = new TreeSet<>(Collections.reverseOrder());
        HashMap<Integer, Double> attNumAndAcc = new HashMap<>();

        for (int i = 0; i < this.ensembleSize; i++) {
            KNN thisKnn = new KNN();
            thisKnn.setStandardise(true);
            thisKnn.setWeighted(false);
            thisKnn.setLOOCV(false);

            Instances training = new Instances(this.train, 0);
            Instances thisInstances, thisTest;

            // --------------- Boostrap aggregation / Bagging ---------------
            for (int j = 0; j < numOfEnsemInstance; j++) {
                ranInt = random.nextInt(numInts);
                intSet.add(ranInt);
                training.add(this.train.instance(ranInt));
            }

            // Construct out of bag data for cheaply estimating the accurarcy
            // i.e. act as a testing set
            Instances outOfBag = new Instances(this.train);
            for (Integer removeIndex : intSet) {
                outOfBag.delete(removeIndex);
            }

            // ------------------ Feature subset selection ------------------
            // If there is only one attribute for the train data
            if (training.numAttributes() == 2) {
                thisKnn.setTrain(training);
                KnnEnsemble.add(thisKnn);
                break;
            }

            // Set initial number of attribute to include as square root of total no. of feature
            int attToInclude = (int) (Math.sqrt(training.numAttributes()));

            // If there are at least 9 attributes in the train data set
            if (attToInclude > 2) {
                // For +1, +2 and -1, -2 attriNum of sqrt(attNum)
                // Test accurarcy using out of bag errors
                // Include attriNum from bagged dataset with highest accurarcy
                for (int j = attToInclude - 2; j < attToInclude + 3; j++) {
                    thisInstances = new Instances(training);
                    thisTest = new Instances(outOfBag);

                    // For attToInclude - 2 to attToInclude + 2, compute accurarcy
                    for (int x = thisInstances.numAttributes() - 1; x > j; x--) {
                        ranInt = random.nextInt(thisInstances.numAttributes() - 1);
                        thisInstances.deleteAttributeAt(ranInt);
                        thisTest.deleteAttributeAt(ranInt);
                    }

                    // Estimating accurarcy
                    thisKnn.buildClassifier(thisInstances, thisTest);
                    double correct = 0.0;
                    for (int y = 0; y < thisKnn.test.numInstances(); y++) {
                        if (thisKnn.classifyInstance(thisKnn.test.get(y)) == thisKnn.test.get(y).classValue()) {
                            correct++;
                        }
                    }
                    attNumAndAcc.put(j, correct / thisKnn.test.numInstances());
                }

                // Find the attribute number that yields the highest accurarcy
                Map<Integer, Double> sorted = attNumAndAcc
                        .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                        .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

                attToInclude = Integer.valueOf(sorted.entrySet().toArray()[0].toString().split("=")[0]);

                // And reconstruct a new train data set using that attribute number
                thisInstances = new Instances(training);
                thisTest = new Instances(outOfBag);
                for (int j = thisInstances.numAttributes() - 1; j > attToInclude; j--) {
                    ranInt = random.nextInt(thisInstances.numAttributes() - 1);
                    thisInstances.deleteAttributeAt(ranInt);
                    thisTest.deleteAttributeAt(ranInt);
                }
                thisKnn.buildClassifier(thisInstances, thisTest);

            } else if (attToInclude == 2) { // At least 4 and less than 9 total features
                // For +1 and -1 attriNum of sqrt(attNum)
                // Test accurarcy using out of bag errors
                // Include attriNum from bagged dataset with highest accurarcy
                for (int j = attToInclude - 1; j < attToInclude + 2; j++) {
                    thisInstances = new Instances(training);
                    thisTest = new Instances(outOfBag);

                    // For attToInclude - 1 to attToInclude + 1, compute accurarcy
                    for (int x = thisInstances.numAttributes() - 1; x > j; x--) {
                        ranInt = random.nextInt(thisInstances.numAttributes() - 1);
                        thisInstances.deleteAttributeAt(ranInt);
                        thisTest.deleteAttributeAt(ranInt);
                    }
                    // estimating accurarcy
                    thisKnn.buildClassifier(thisInstances, thisTest);
                    double correct = 0.0;
                    for (int y = 0; y < thisKnn.test.numInstances(); y++) {
                        if (thisKnn.classifyInstance(thisKnn.test.get(y)) == thisKnn.test.get(y).classValue()) {
                            correct++;
                        }
                    }
                    attNumAndAcc.put(j, correct / thisKnn.test.numInstances());

                    // Find the attribute number that yields the highest accurarcy
                    Map<Integer, Double> sorted = attNumAndAcc
                            .entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                            .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));
                    attToInclude = Integer.valueOf(sorted.entrySet().toArray()[0].toString().split("=")[0]);

                    // And reconstruct a new train and test data set using that attribute number
                    thisInstances = new Instances(training);
                    thisTest = new Instances(this.test);
                    for (int n = thisInstances.numAttributes() - 1; n > attToInclude; n--) {
                        ranInt = random.nextInt(thisInstances.numAttributes() - 1);
                        thisInstances.deleteAttributeAt(ranInt);
                        thisTest.deleteAttributeAt(ranInt);
                    }
                    thisKnn.buildClassifier(thisInstances, thisTest);
                    if (thisKnn.useLOOCV) {
                        thisKnn.LOOCV();
                    }

                }
            } else { // 2 or 3 total features
                thisInstances = new Instances(training);
                thisTest = new Instances(this.test);
                for (int m = training.numAttributes() - 1; m > attToInclude; m--) {
                    ranInt = random.nextInt(thisInstances.numAttributes() - 1);
                    thisInstances.deleteAttributeAt(ranInt);
                    thisTest.deleteAttributeAt(ranInt);
                }
                thisKnn.buildClassifier(thisInstances, thisTest);
                if (thisKnn.useLOOCV) {
                    thisKnn.LOOCV();
                }
            }
            intSet.clear();
            KnnEnsemble.add(thisKnn);
        }
        return KnnEnsemble;
    }
    
    /**
     * 
     * @return the size of the ensemble
     */
    public int getEnsembleSize() {
        return ensembleSize;
    }

    /**
     * 
     * @param ensembleSize to set the size of the ensemble
     */
    public void setEnsembleSize(int ensembleSize) {
        this.ensembleSize = ensembleSize;
    }

    /**
     * 
     * @return to get the ensemble of knn classifiers as an array list
     */
    public ArrayList<KNN> getEnsemble() {
        return ensemble;
    }

    /**
     * 
     * @param ensemble to set the ensemble with an array list of kNN classifiers
     */
    public void setEnsemble(ArrayList<KNN> ensemble) {
        this.ensemble = ensemble;
    }

}
