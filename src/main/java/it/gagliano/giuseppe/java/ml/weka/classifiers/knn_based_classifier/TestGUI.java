package it.gagliano.giuseppe.java.ml.weka.classifiers.knn_based_classifier;

import java.awt.EventQueue;
import java.awt.Font;
import java.awt.SystemColor;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.SwingConstants;

public class TestGUI extends JPanel{

	private JFrame frame;
	private JTextField textFieldKnn;
	private File lastDir;
	private JList<String> datasetJList;

	// Parameters to send
	private ArrayList<String> directoryList;
	private int[] kNN;
	private JProgressBar progressBar;
	private JTextPane progressStatus;
	public static JButton btnStart, btnCancel, btnShowResult;
	private TestClass programThread;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					TestGUI window = new TestGUI();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public TestGUI() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 480, 290);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
		
		// ***************** DATASET ********************
		JLabel lblDatasets = new JLabel("Datasets");
		lblDatasets.setFont(new Font("Tahoma", Font.PLAIN, 11));
		lblDatasets.setBounds(10, 11, 61, 14);
		frame.getContentPane().add(lblDatasets);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(10, 36, 156, 176);
		frame.getContentPane().add(scrollPane);
		datasetJList = new JList<String>(new DefaultListModel<String>());
		scrollPane.setViewportView(datasetJList);

		JButton btnImportDatasets = new JButton("Import Datasets");
		btnImportDatasets.setBounds(10, 223, 156, 23);
		frame.getContentPane().add(btnImportDatasets);
		progressBar = new JProgressBar();
		progressBar.setBounds(195, 232, 259, 14);
		frame.getContentPane().add(progressBar);
		lastDir = new File(System.getProperty("user.dir")+"/data");
		btnImportDatasets.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				File[] theDir = null;
				theDir = selectDir();
				if(theDir != null) {
					directoryList = new ArrayList<String>();
					((DefaultListModel)datasetJList.getModel()).clear();
					for(File z : theDir) {
						String curRow = z.toString();
						directoryList.add(curRow);
						((DefaultListModel)datasetJList.getModel()).addElement(curRow);
					}
				}
				return;
			}
		});

		JSeparator separator = new JSeparator();
		separator.setOrientation(SwingConstants.VERTICAL);
		separator.setBounds(176, 11, 2, 235);
		frame.getContentPane().add(separator);

		JLabel lblkNN = new JLabel("Enter values for k");
		lblkNN.setToolTipText("For multiple values, separate with commas each value. E.g. 1,3,5,10");
		lblkNN.setBounds(195, 11, 181, 14);
		frame.getContentPane().add(lblkNN);

		textFieldKnn = new JTextField();
		textFieldKnn.setText("1,5,10");
		textFieldKnn.setToolTipText("");
		textFieldKnn.setBounds(195, 35, 86, 20);
		frame.getContentPane().add(textFieldKnn);
		textFieldKnn.setColumns(10);

		progressStatus = new JTextPane();
		progressStatus.setBorder(javax.swing.BorderFactory.createEmptyBorder());
		progressStatus.setEditable(false);
		progressStatus.setForeground(SystemColor.windowText);
		progressStatus.setBackground(SystemColor.menu);
		progressStatus.setText("Nothing to show");
		progressStatus.setFont(new Font("Tahoma", Font.PLAIN, 11));
		progressStatus.setBounds(195, 102, 160, 119);
		frame.getContentPane().add(progressStatus);

		btnStart = new JButton("Start");
		btnStart.setBounds(365, 34, 89, 23);
		frame.getContentPane().add(btnStart);
		btnStart.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent arg0) {
				if(directoryList == null)	JOptionPane.showMessageDialog(frame, "You need to import datasets before starting.");
				else{
					String[] txt = textFieldKnn.getText().replace(" ", "").split(",");
					kNN = new int[txt.length];
					for(int i=0;i<txt.length;i++)	kNN[i] = Integer.parseInt(txt[i]);	
					programThread = new TestClass(kNN, progressBar, progressStatus, directoryList);
					programThread.start();
					btnStart.setEnabled(false);
					btnCancel.setEnabled(true);
				}
			}
		});

		btnCancel = new JButton("Stop");
		btnCancel.setBounds(365, 68, 89, 23);
		btnCancel.setEnabled(false);
		frame.getContentPane().add(btnCancel);
		btnCancel.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent arg0) {
				programThread.stop();
				btnCancel.setEnabled(false);
				btnStart.setEnabled(true);

			}
		});

		btnShowResult = new JButton("Results...");
		btnShowResult.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				try {
					Runtime.getRuntime().exec("explorer "+System.getProperty("user.dir")+"\\out");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		});
		btnShowResult.setBounds(365, 200, 89, 23);
		frame.getContentPane().add(btnShowResult);

		JLabel lblComputationStatus = new JLabel("Status");
		lblComputationStatus.setBounds(195, 77, 163, 14);
		frame.getContentPane().add(lblComputationStatus);

	}

	private File[] selectDir() {
		JFileChooser fileChooser = new JFileChooser(lastDir);
		fileChooser.setMultiSelectionEnabled(true);
		fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		int showOpenDialog = fileChooser.showOpenDialog(null);
		if (showOpenDialog != JFileChooser.APPROVE_OPTION) {
			return null;
		}
		File[] uploadDir = fileChooser.getSelectedFiles();
		lastDir = new File(uploadDir[uploadDir.length-1].getParent());
		return uploadDir;
	}
}
