package it.gagliano.giuseppe.java.ml.weka.classifiers.knn_based_classifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;

import javax.swing.JProgressBar;
import javax.swing.JTextPane;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class TestClass extends Thread{

	// PARAMETERS TO SET
	private final static int K_FOLD = 10;
	private static int[] KNN;
	private static ArrayList<String> datasetPaths;

	// Variables
	private enum TAGS {ibk,sco,lin};
	private static PrintWriter [] writer;
	private static PrintWriter [] writerDuration;
	private static PrintWriter writerMeans;
	private static double [] meanTrainingAccuracy;
	private static double [] meanTestingAccuracy;

	// Update current status
	private static JProgressBar progressBar;
	private static JTextPane progressStatus;

	public TestClass() {
		super();
		// TODO Auto-generated constructor stub
	}

	public TestClass(int[] kNN, JProgressBar progressBar, JTextPane progressStatus, ArrayList<String> datasetPaths) {
		KNN = kNN;
		progressBar.setMinimum(1);
		progressBar.setMaximum(100);
		progressBar.setValue(1);
		TestClass.progressBar = progressBar;
		TestClass.progressStatus = progressStatus;
		TestClass.datasetPaths = datasetPaths;
	}

	public static void main(String[] args) throws Exception {
		(new TestClass()).start();
	}

	public static void evaluateClassifier(TAGS tag, Instances trainingSet, Instances testSet, int foldNum, int knn) throws Exception{

		// Initialize the right classifier
		Classifier classifier;
		long durationBuilding;
		long durationClassification;
		double accuracyTraining = 0.0, accuracyTesting = 0.0;
		switch(tag){
		case sco:	classifier = new KNNGagliano(true, knn);	break;
		case lin:	classifier = new KNNGagliano(false, knn);	break;
		default:	classifier = new IBk(knn);
		}

		// Build and evaluate
		classifier.buildClassifier(trainingSet);
		Evaluation eval = new Evaluation(trainingSet);
		eval.evaluateModel(classifier, trainingSet);
		accuracyTraining = eval.pctCorrect();
		durationBuilding = System.nanoTime();
		eval = new Evaluation(trainingSet);
		durationBuilding = System.nanoTime() - durationBuilding;
		durationClassification = System.nanoTime();
		eval.evaluateModel(classifier, testSet);
		durationClassification = System.nanoTime() - durationClassification;
		accuracyTesting = eval.pctCorrect();

		// Accumulate means
		meanTrainingAccuracy[tag.ordinal()] += accuracyTraining;
		meanTestingAccuracy[tag.ordinal()] += accuracyTesting;

		// Write to a file
		writer[tag.ordinal()].println(String.valueOf(foldNum)+"\t\t"+
				String.valueOf(accuracyTraining)+"\t\t"+
				String.valueOf(accuracyTesting));
		writerDuration[tag.ordinal()].println(String.valueOf(foldNum)+"\t\t"+
				String.valueOf(durationBuilding)+"\t\t"+
				String.valueOf(durationClassification));
	}

	@Override
	public void run() {
		// Initialize variables
		meanTrainingAccuracy = new double[TAGS.values().length];
		meanTestingAccuracy = new double[TAGS.values().length];
		int count = 1;
		ArffLoader al = new ArffLoader();

		try{			
			// Initialize file writers
			writer = new PrintWriter[TAGS.values().length];
			writerDuration = new PrintWriter[TAGS.values().length];
			for(int h=0;h<KNN.length;h++){
				if (!(new File("out")).exists())	(new File("out")).mkdir();
				if (!(new File("out/knn"+KNN[h])).exists())	(new File("out/knn"+KNN[h])).mkdir();
				if (!(new File("out/knn"+KNN[h]+"/accuracy/")).exists())	(new File("out/knn"+KNN[h]+"/accuracy")).mkdir();
				if (!(new File("out/knn"+KNN[h]+"/duration/")).exists())	(new File("out/knn"+KNN[h]+"/duration")).mkdir();
				writerMeans = new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/accuracy_means.txt")),true);
				writerMeans.close();
				writerMeans= new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/accuracy_means.txt"),true),true);
				for(TAGS tag:TAGS.values()){
					writer[tag.ordinal()] = new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/accuracy/"+tag+"_accuracy.txt")),true);
					writer[tag.ordinal()].close();
					writer[tag.ordinal()] = new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/accuracy/"+tag+"_accuracy.txt"),true),true);
					writerDuration[tag.ordinal()] = new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/duration/"+tag+"_duration.txt")),true);
					writerDuration[tag.ordinal()].close();
					writerDuration[tag.ordinal()] = new PrintWriter(new FileOutputStream(new File("out/knn"+KNN[h]+"/duration/"+tag+"_duration.txt"),true),true);
				}

				// Importing Datasets
				ArrayList<File> listOfDataset = new ArrayList<File>();
				for(String path:datasetPaths) listOfDataset.add(new File(path));

				// For each dataset
				for (int i = 0; i < listOfDataset.size(); i++) {
					// Initialize some variables
					for(TAGS tag:TAGS.values())	{
						meanTrainingAccuracy[tag.ordinal()] = 0.0;
						meanTestingAccuracy[tag.ordinal()] = 0.0;
					}
					String datasetPath = "data/"+listOfDataset.get(i).getName()+ "/"+
							listOfDataset.get(i).getName().substring(0,listOfDataset.get(i).getName().indexOf("fold"));

					// Set labels for the statistic files
					for(File f:listOfDataset.get(i).listFiles())
						if(f.getPath().endsWith(".arff")){
							al.setFile(new File(f.getPath()));
							break;
						}
					int numAttr = al.getDataSet().numAttributes();
					writerMeans.println("************************* Dataset "+
							listOfDataset.get(i).getName().substring(0,listOfDataset.get(i).getName().indexOf("-fold"))+ " ("+
							numAttr +" attributes) *************************\nclassifier\t\tmean(training)\t\tmean(testing)");
					for(TAGS tag:TAGS.values()){
						writer[tag.ordinal()].println("***************************** Dataset "+
								listOfDataset.get(i).getName().substring(0,listOfDataset.get(i).getName().indexOf("-fold"))+ " ("+
								numAttr +" attributes) *************************\n# of fold\t\tacc(training)\t\tacc(testing)");
						writerDuration[tag.ordinal()].println("*************************** Dataset "+
								listOfDataset.get(i).getName().substring(0,listOfDataset.get(i).getName().indexOf("-fold"))+ " ("+
								numAttr +" attributes) *************************\nfold\t\tbuild\t\tclassification[ms]");
					}

					// For each of the k-fold data partitioning
					for(int j=1; j<K_FOLD+1; j++){
						al.setFile(new File(datasetPath+String.valueOf(j)+"tra.arff"));
						Instances trainingSet = al.getDataSet();
						trainingSet.setClassIndex(trainingSet.numAttributes()-1);
						al.setFile(new File(datasetPath+String.valueOf(j)+"tst.arff"));
						Instances testSet = al.getDataSet();
						testSet.setClassIndex(testSet.numAttributes()-1);

						// Evaluate Classifiers
						progressStatus.setText("kNN: "+KNN[h]+"\ndataset: " +
								datasetPath.toString().split("/")[2].split("-")[0] +
								"\nfold: " + j + "\n" +
								(count*100)/(KNN.length*datasetPaths.size()*K_FOLD)+
								"%");
						progressBar.setValue((count*100)/(KNN.length*datasetPaths.size()*K_FOLD));
						count ++;
						for(TAGS tag:TAGS.values()){
							evaluateClassifier(tag, trainingSet, testSet, j, KNN[h]);
						}	
					}

					// Update means for each dataset and write them to file
					for(TAGS tag:TAGS.values()){
						meanTestingAccuracy[tag.ordinal()] /= K_FOLD;
						meanTrainingAccuracy[tag.ordinal()] /= K_FOLD;
						writer[tag.ordinal()].println("mean \t\t"+
								String.valueOf(meanTrainingAccuracy[tag.ordinal()])+"\t\t"+
								String.valueOf(meanTestingAccuracy[tag.ordinal()]));
						writerMeans.println(tag+"\t\t"+
								String.valueOf(meanTrainingAccuracy[tag.ordinal()])+"\t\t"+
								String.valueOf(meanTestingAccuracy[tag.ordinal()]));
					}
				}
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally{
			progressStatus.setText("Completed.");
			TestGUI.btnCancel.setEnabled(false);
			TestGUI.btnStart.setEnabled(true);
			for(PrintWriter w:writer){
				w.flush();
				w.close();
			}
			writerMeans.flush();
			writerMeans.close();
		}

	}

}
