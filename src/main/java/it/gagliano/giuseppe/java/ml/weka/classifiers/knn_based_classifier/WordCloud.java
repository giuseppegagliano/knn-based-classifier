package it.gagliano.giuseppe.java.ml.weka.classifiers.knn_based_classifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import weka.core.converters.ArffLoader;

public class WordCloud {

	private static PrintWriter writer;

	public static void main(String[] args) throws IOException {
		File dataFolder = new File("data/");
		File[] listOfFolders = dataFolder.listFiles();
		int instanceCounter;
		int totalInstances;
		int maxInstances;
		int testInstances;

		try{
			writer = new PrintWriter(new FileOutputStream(new File("out/wordCloud.txt"),true),true);

			// For each dataset
			for (int i = 0; i < listOfFolders.length; i++) {
				if (listOfFolders[i].isDirectory()) {
					instanceCounter = 0;
					testInstances = 0;
					String datasetPath = "data/"+listOfFolders[i].getName()+ "/"+
							listOfFolders[i].getName().substring(0,listOfFolders[i].getName().indexOf("fold"));
					ArffLoader al = new ArffLoader();
					al.setFile(new File(datasetPath+"1tra.arff"));
					instanceCounter += al.getDataSet().size();
//					System.out.println(instanceCounter);
					al.setFile(new File(datasetPath+"1tst.arff"));
					testInstances = al.getDataSet().size();
					instanceCounter += testInstances;
//					System.out.println(testInstances);
					for(int j=0;j<(int)(instanceCounter/80);j++){
						writer.println(listOfFolders[i].getName().substring(0,listOfFolders[i].getName().indexOf("-10")));
					}
				}
			}
		} finally{
			writer.close();
		}
	}

}
