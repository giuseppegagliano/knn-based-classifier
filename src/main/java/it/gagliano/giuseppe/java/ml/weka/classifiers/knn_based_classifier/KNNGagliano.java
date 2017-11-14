package it.gagliano.giuseppe.java.ml.weka.classifiers.knn_based_classifier;

import java.text.DecimalFormat;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class KNNGagliano extends AbstractClassifier implements weka.classifiers.Classifier{

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	/** The training instances used for classification. */
	protected Instances m_Train;

	/** The number of class values (or 1 if predicting numeric). */
	protected int m_NumClasses;

	/** The class attribute type. */
	protected int m_ClassType;

	/** The number of neighbours to use for classification (currently). */
	protected int m_kNN;

	/**
	 * The value of kNN provided by the user. This may differ from
	 * m_kNN if cross-validation is being used.
	 */
	protected int m_kNNUpper;

	/**
	 * Whether the value of k selected by cross validation has
	 * been invalidated by a change in the training instances.
	 */
	protected boolean m_kNNValid;

	/**
	 * The maximum number of training instances allowed. When
	 * this limit is reached, old training instances are removed,
	 * so the training data is "windowed". Set to 0 for unlimited
	 * numbers of instances.
	 */
	protected int m_WindowSize;

	/** Whether the neighbours should be distance-weighted. */
	protected int m_DistanceWeighting;

	/** Whether to select k by cross validation. */
	protected boolean m_CrossValidate;

	/**
	 * Whether to minimise mean squared error rather than mean absolute
	 * error when cross-validating on numeric prediction tasks.
	 */
	protected boolean m_MeanSquared;

	/** Default ZeroR model to use when there are no training instances */
	protected ZeroR m_defaultModel;

	/** for nearest-neighbor search. */
	protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

	/** The number of attributes the contribute to a prediction. */
	protected double m_NumAttributesUsed;

	/** The score matrix where the element i-j contains the score obtained
	 * by the i-th attribute with respect to the j-th class computed as
	 * score = Nk/k
	 * */
	protected Double [] [] m_ScoreMatrix;

	/** The array containing the number of times that a class won while
	 * classifying an instance by single attribute
	 * */
	protected int [] m_ClassesFrequency;

	/*
	 * Whether to use the first classifier or the second.
	 * In the first classifier each feature from the instance is classified
	 * and then the instance class is that one having greater number of 
	 * classified feature.
	 * In the second each feature belongs to each class with a certain weight,
	 * at the end, weights are summed up by class, the class with the maximum
	 * weight is the estimated class
	 * */
	boolean m_IsSco;


	/************************MANDATORY METHODS***************************/

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @throws Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		// init variables
		m_NumClasses = instances.numClasses();
		m_ClassType = instances.classAttribute().type();
		m_Train = new Instances(instances, 0, instances.numInstances());

		// Throw away initial instances until within the specified window size
		if ((m_WindowSize > 0) && (instances.numInstances() > m_WindowSize)) {
			m_Train = new Instances(m_Train, 
					m_Train.numInstances()-m_WindowSize, 
					m_WindowSize);
		}

		// remove class attribute from the training set
		m_NumAttributesUsed = 0.0;
		for (int i = 0; i < m_Train.numAttributes(); i++) {
			if ((i != m_Train.classIndex()) && 
					(m_Train.attribute(i).isNominal() ||
							m_Train.attribute(i).isNumeric())) {
				m_NumAttributesUsed += 1.0;
			}
		}

		// set the training set as input data for the nn search class
		m_NNSearch.setInstances(m_Train);

		// Invalidate any currently cross-validation selected k
		m_kNNValid = false;

		/**
		 * build a ZeroR classifier which predicts only the majority class
		 * used in distributionForInstance
		 * ZeroR is the simplest classification method which relies on the 
		 * target and ignores all predictors. ZeroR classifier simply predicts
		 * the majority category (class). Although there is no predictability
		 * power in ZeroR, it is useful for determining a baseline performance
		 *  as a benchmark for other classification methods.
		 */

		m_defaultModel = new ZeroR();
		m_defaultModel.buildClassifier(instances);
	}

	/**
	 * Returns the membership class for the specified instance.
	 *
	 * @return      the membership class for the specified instance
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		
		// Initialization
		boolean onDebug = false;
		m_ScoreMatrix = new Double [instance.numAttributes()-1][m_NumClasses];
		Double maxScore = 0.0;
		int winningClassIndex = 0;
		m_ClassesFrequency = new int [m_NumClasses];
		for (int i = 0; i < m_ClassesFrequency.length; i++) {m_ClassesFrequency[i] = 0;}
		Instances attr_kNN;

		// For each feature (class attribute excluded)
		for(int i=0;i<instance.numAttributes()-1;i++){

			// Compute distance only using the i-th attribute
			m_NNSearch.getDistanceFunction().setAttributeIndices(String.valueOf(i+1));

			// Distances are normalized, to disable normalizing use the following line
			((EuclideanDistance)m_NNSearch.getDistanceFunction()).setDontNormalize(true);

			// Due to neighbors with the same distance there may be more than k neighbors
			attr_kNN = m_NNSearch.kNearestNeighbours(instance, m_kNN);

			//	(Debug) Show distances and neighbors
			if(onDebug){
				System.out.println("distance on attribute "+m_NNSearch.getDistanceFunction().getAttributeIndices().toString());
				for(Instance nn:attr_kNN){
					System.out.println("Distance: "+String.valueOf(instance.value(i))+" - "+
							String.valueOf(nn.value(i))+"\t\t="+
							String.valueOf(m_NNSearch.getDistanceFunction().distance(instance, nn))+
							" (not normalized is "+String.valueOf(Math.abs(instance.value(i)-nn.value(i)))+")"+
							"\t neighbor class: "+nn.classValue());
				}
			}

			// Build the score matrix and store frequencies
			for(int j=0;j<m_NumClasses;j++){
				if(onDebug)				System.out.println("Estimated Nk for class "+j+" is "+String.valueOf(((double)computeNk(attr_kNN,j))));
				m_ScoreMatrix [i][j] = (((double)computeNk(attr_kNN,j))/Math.max(m_kNN,attr_kNN.size()));
				if((j==0)||(m_ScoreMatrix[i][j]>maxScore)){
					maxScore = m_ScoreMatrix[i][j];
					winningClassIndex = j;
				}
			}
			m_ClassesFrequency[winningClassIndex]++;
		}

		if(m_IsSco)	return classifierSco(instance);
		return classifierLin(instance);
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		result.setMinimumNumberInstances(0);

		return result;
	}


	/**************************USEFUL OR RELATED METHODS********************/

	/* Initialize some attributes, maybe will be added an init() method */
	public KNNGagliano() {
		setKNN(1);
		m_WindowSize = 0;
		m_CrossValidate = false;
		m_MeanSquared = false;
		m_IsSco = false;
	}
	
	/* Initialize some attributes, gets the boolean to specify the classifier to use  */
	public KNNGagliano(boolean IsScoClassifier) {
		setKNN(1);
		m_WindowSize = 0;
		m_CrossValidate = false;
		m_MeanSquared = false;
		m_IsSco = IsScoClassifier;
	}
	
	/* Initialize some attributes, gets the boolean to specify the classifier to use
	 * and the number of neighbors  */
	public KNNGagliano(boolean IsScoClassifier, int kNN) {
		setKNN(kNN);
		m_WindowSize = 0;
		m_CrossValidate = false;
		m_MeanSquared = false;
		m_IsSco = IsScoClassifier;
	}

	/**
	 * Set the number of neighbours the learner is to use.
	 *
	 * @param k the number of neighbours.
	 */
	public void setKNN(int k) {
		m_kNN = k;
		m_kNNUpper = k;
		m_kNNValid = false;
	}

	/**
	 * Gets the number of neighbours the learner will use.
	 *
	 * @return the number of neighbours.
	 */
	public int getKNN() {

		return m_kNN;
	}

	/**
	 * Adds the supplied instance to the training set.
	 *
	 * @param instance the instance to add
	 * @throws Exception if instance could not be incorporated
	 * successfully
	 */
	public void updateClassifier(Instance instance) throws Exception {

		if (m_Train.equalHeaders(instance.dataset()) == false) {
			throw new Exception("Incompatible instance types\n" + m_Train.equalHeadersMsg(instance.dataset()));
		}
		if (instance.classIsMissing()) {
			return;
		}

		m_Train.add(instance);
		m_NNSearch.update(instance);
		m_kNNValid = false;
		if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
			boolean deletedInstance=false;
			while (m_Train.numInstances() > m_WindowSize) {
				m_Train.delete(0);
				deletedInstance=true;
			}
			//rebuild datastructure KDTree currently can't delete
			if(deletedInstance==true)
				m_NNSearch.setInstances(m_Train);
		}
	}

	/**
	 * Gets the number of neighbours with a specified class index.
	 *
	 * @return the number of neighbours with a specified class index.
	 */
	public int computeNk(Instances neighbours, double classValue) {
		int count = 0;

		for(Instance neighbour:neighbours){
			if(neighbour.classValue() == classValue)
				count++;
		}
		return count;
	}

	/**
	 * Prints score matrix and winning class frequencies
	 *
	 * @return a string version of the score matrix
	 */
	public void printScores(Instance instance){
		StringBuilder sb = new StringBuilder("**************************************Scores Matrix******************************\n");
		sb.append("Feat\t\t\t");
		for(int k=0;k<m_NumClasses;k++)	sb.append("Class "+k+"\t\t\t");
		sb.append("\n");
		for(int i=0;i<instance.numAttributes()-1;i++){
			sb.append(i+"\t\t\t");
			for(int j=0;j<m_NumClasses;j++){
				DecimalFormat df = new DecimalFormat();
				df.setMaximumFractionDigits(3);
				sb.append(df.format(m_ScoreMatrix[i][j])+"\t\t\t");
			}
			sb.append("\n");
		}
		System.out.println(sb.toString());


		System.out.println("******************************Classes frequency***********************************");
		System.out.println("class\t\tfreq");
		for(int k=0;k<m_NumClasses;k++)	System.out.println(k+"\t\t"+m_ClassesFrequency[k]);
		System.out.println();
	}

	/**
	 * In the first classifier each feature from the instance is classified
	 * and then the instance class is that one having greater number of 
	 * classified feature.
	 * 
	 * @return the estimated class
	 */
	public double classifierSco(Instance instance){
		// m_ClassesFrequency counts how many attributes were classified with the i-th class
		int estimatedClass = 0, freq = m_ClassesFrequency[0];

		for(int i=0;i<m_NumClasses;i++){
			if(m_ClassesFrequency[i]>freq){
				estimatedClass = i;
				freq = m_ClassesFrequency[i];
			}
		}
		return (estimatedClass);
	}

	/**
	 * In the second each feature belongs to each class with a certain weight,
	 * at the end, weights are summed up by class, the class with the maximum
	 * weight is the estimated class
	 *
	 * @return the estimated class
	 */
	public double classifierLin(Instance instance){
		// Init
		Double [] classScores = new Double[m_NumClasses];
		double maxScore = 0.0;
		for(int j=0;j<m_NumClasses;j++)	classScores[j]=0.0;

		// Sum scores by class
		for(int i=0;i<instance.numAttributes()-1;i++)
			for(int j=0;j<m_NumClasses;j++)
				classScores[j]+=m_ScoreMatrix[i][j];

		// Compute max score class
		int maxScoreClassIndex=0;
		maxScore = classScores[0];
		for(int j=1;j<m_NumClasses;j++)
			if(classScores[j]>maxScore){
				maxScore = classScores[j];
				maxScoreClassIndex = j;
			}

		return (maxScoreClassIndex);
	}

	/*******************************MAIN*****************************************/
	public static void main(String[] args) { 
		runClassifier(new KNNGagliano(), args); 
	} 

}
