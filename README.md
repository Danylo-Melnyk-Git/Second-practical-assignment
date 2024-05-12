This practical assignment aims to develop students' skills in using machine learning algorithms and analysing the results. To complete the assignment, the student team must select a dataset and use supervised and unsupervised machine learning algorithms to process it. 

Students are advised to use the Orange tool to develop their assignments. Its user manual is available on the next page. The following Orange widgets are of particular value in the context of the assignment: File, Data Table, Data Sampler, Bar Plot, Scatter plot, Feature Statistics, Distributions, Test and Score, Predictions, Confusion matrix, Silhouette plot, Roc analysis, as well as widgets for various machine learning algorithms. At the same time, the student team can choose to complete the assignment in Python. The following task description is mainly attributed to the Orange tool, but the same requirements apply if the student team uses Python.

To complete the assignment, students may need to independently search and research additional sources of information in order to answer the questions of the assignment or to provide analysis and interpretation of the results.
Use the following link to download and install the Orange tool:
https://orangedatamining.com/download/#windows

To find more information about the tool and its features, use:

    Tool website
    Tutorials available on YouTube 

In any case, be curious! Don't be afraid to experiment with the tool!


Creating a project
You need to create a new project to get started with the Orange tool. It can be done in one of two ways:

    by selecting "New" on the tool's welcome screen;
    using the "File/New" menu or the Ctrl+N key combination.

The main working principles of the tool

To use the Orange tool, you need to know two key concepts:

    a widget;
    a data analysis workflow.

Widgets are the main data processing units of the tool. They allow you to do the following: read and display data, visualise data, create a model, check the performance of the model, etc. There are five main categories of widgets. They are located on the left side of the screen and are as follows:

    Data – widgets for data manipulation;
    Visualise – widgets for data visualisation; 
    Model – widgets for making predictions (supervised learning);
    Evaluate – widgets for evaluating model performance;
    Unsupervised – widget for unsupervised learning.

Widgets communicate with each other. The left-hand side of each widget is its input channel, and the right-hand side is its output channel. Some widgets may have only an input channel. You open widgets by double-clicking on them. Afterwards, you can change the widget's settings or observe data/results provided by the widget.

Data analysis workflows are based on the relationships between widgets. Relationships serve as communication channels. To relate widgets, drag the cursor from the right side of one widget (output channel) to the left side of another widget (input channel). The Orange tool does not allow relating incompatible widgets. Workflows allow data to flow from one widget to another, and changes in a widget are immediately propagated through the workflow.

Data loading

You need to load data to start using data analysis algorithms and visualisation methods. To do this, click on the "File" widget in the "Data" category on the left-hand side of the tool, and it will appear on the canvas. You can then change the name of the widget, e.g. "My training dataset", by pressing the F2 button. This widget allows you to load a dataset (test or training dataset, depending on the purpose).

After adding the widget to the canvas, you can load the training dataset into the previously added file widget. Open the file widget by double-clicking and select a data source: File - to load data from a file stored on your computer; URL - to load data from a link if your data is stored somewhere outside your computer. Click the "Reload" button. You will see information about the dataset in the "Info" field. The "Columns" field will display the columns of the dataset, and you can also change the role of any feature (attribute) (see the "Role" column). Check that the correct column in your dataset is selected as the target column (contains class labels). Click the "Apply" button and close the window.

Another data processing widget that is used quite often is the Data Table. It allows you to display data in a spreadsheet. Click on this widget under the "Data" category, and the widget will appear on the canvas. The file widget must then be related to the data table widget by dragging the cursor from the right-hand side of the file widget to the left-hand side of the data table widget.

Example of a workflow

Let's create an example of data analysis workflow using the following steps:

    Add the "Distributions" widget from the "Visualise" category to the canvas and connect the data table widget to it. Double-click the "Distributions" widget to check the distribution of different features (attributes) in your training dataset.
    Add the "Scatter Plot" widget from the "Visualise" category to the canvas and connect the data table widget to it. Then, double-click the "Scatter Plot" widget to view the relationships between the different features (attributes).
    Add a kNN widget from the "Model" category to the canvas and connect the file widget to it. Double-click the kNN widget to change the classifier settings.
    Add another "File" widget to the canvas, change its title to "My testing dataset" and load the data from the test dataset.
    Add a "Predictions" widget from the "Evaluate" category and connect the kNN widget and the file widget "My testing dataset" to it. Then double-click the "Predictions" widget to view the performance of the kNN classifier on the test dataset.

To find a dataset for the assignment, the student team can use the following well-known repositories:

    UC Irvine Machine Learning Repository 
    R Datasets on Github 
    Kaggle Datasets 
    Yahoo! Webscope Datasets 
    Reddit

When selecting a dataset, the student team must consider the following:

    A dataset that is suitable for the classification task should be selected. The values of the output variable must be categorical to implement the classification task. The difference between categorical and continuous values is described at the end of this page.
    The student team must select something other than the Iris flower dataset or the Palmer Archipelago (Antarctica) penguin dataset. 
    When choosing a dataset, the meaningfulness of the classification should be considered, e.g., a) it does not make sense to classify continents according to the number of Covid-19 events because, first, there are only six continents and no new ones will appear soon, and, second, the number of Covid-19 events is not a characteristic of a continent; b) it is meaningless to classify fuel types if the dataset is about car characteristics such as fuel type, kilometres travelled, engine capacity, etc. These are characteristics of cars, not fuel types; c) it is meaningless to classify parents according to their level of education if the data are on the grades of children in different subjects.
    It is preferable to choose a dataset that is already given in a .csv data file format.
    The dataset must be well documented (information on the creator of the dataset, the time when it was created and the source of the data should be available).
    The dataset must be reasonably sized (at least 200 data objects). A large number of data objects and/or features may: a) cause difficulties in describing and visually interpreting the dataset; and b) cause problems in processing the dataset, as your personal computer may need more computing resources.
    The number of data features (attributes) should be between 5 and 15.
    The dataset must contain a detailed description of the data features (attributes) and their meaning.   
    The dataset must contain class labels.
    Students must avoid datasets that contain many Boolean type (true/false, 1/0, etc.) or categorical type feature (attribute) values. Using datasets where most of the features are represented by continuous feature values is preferable. If the dataset contains mainly categorical values, the results will be less relevant as the number of categorical values is often limited. Meaningful relationships in the data can only be discovered if the dataset contains mostly continuous feature values, having a wide variety of ranges and values.  
    Students must avoid datasets of unlabelled data (e.g. text corpora and raw images).

Categorical values 	Continuous values
They represent discrete values corresponding to categories:
- Animal: dog, cat, horse, snake, etc.
- Gender: male and female
- Color: green, blue, white, etc.   	They represent numerical values within a defined range:
- Temperature [0, 100]
- House price [100000, 5 000 000]
In the dataset, they can be represented as:
- symbol strings (dog, cat, horse, snake, etc.)
- numerical values, when each specific category is labelled by a specific numerical value, for example, dog - 1, cat - 2, horse- 3, snake - 4, etc.    	They are represented only as numerical values
They cannot be compared using mathematical operations even is the categories are represented numerically:
- If categories are represented as symbol strings (dog, cat, horse, snake, etc.), it is impossible to say that cat is greater than dog.
- If categories are represented numerically like dog - 1, cat - 2, horse- 3, snake - 4, they still cannot be compared and it is impossible to say that 2 is greater than 1, because 2 represents cats but 1 - dogs. 	They can be compared using mathematical operations. It is possible to say that 4 is greater than 3 or 27 is smaller than 45
They do not have decimal values, if the categories are represented numerically 	They can have decimals
 
The general requirements that the student team should follow when reporting on the practical assignment are the following:

    The report must contain the following parts: a title page, a page with a visual representation of the Orange tool workflow for the whole project, sections corresponding to the parts of the assignment (I, II and III), and a list of information sources used.
    On the title page, the student team must give a link to the project and dataset on a public site (e.g. Google Drive, GitHub, etc.). The teacher must be able to download the team's project without additional registration and restrictions using the provided link.
    In all three parts of the report, the student team must clearly describe if something is irrelevant to their work, e.g. there is no information on licensing aspects of the dataset, or the algorithm does not have hyperparameters. 
    All three parts of the assignment must be accompanied by evidence of the work done, i.e. screenshots demonstrating the Orange widget settings and the results obtained. 
    The figures and tables used in the report must be numbered, and their explanations and references should be given in the text.
    The report should be submitted as a single .docx or .pdf file. 
    The report must not be supplemented with unnecessary theory and additional information. The student team must give concise answers.

Data pre-processing/exploring

Goal: this part aims to familiarise the student team with the dataset, explore its characteristics and identify the features that will be used in the classification task.

The student team is required to do the following to complete this part of the assignment:

    Select and describe the dataset based on the information provided in the repository where the dataset is available.
    If the dataset retrieved from the repository is not in a format that is easy to work with (e.g. comma, comma-separated values or .csv file), transform it into the required format. 
    If the values of any features (attributes) are textual values (e.g. yes/no, positive/neutral/negative, etc.), transform them into numeric values.
    If some data objects have missing or outlier values for some features (attributes), find a way to deal with them by studying additional sources of information.
    Represent the dataset visually and calculate statistical indicators:

    create at least two 2- or 3-dimensional scatter plots illustrating class separability based on different features (attributes); students should avoid using the data object ID or class feature as a variable in the scatter plots;
    create at least 2 histograms showing the separation of classes based on features (attributes) of interest;
    show the distribution of the 2 features (attributes) of interest;
    calculate statistical indicators (at least the central tendency and the dispersion of the feature values).

Include the following information in the report:
description of the dataset (including references to the information sources used) 	- title, source, creator and/or owner of the dataset
- description of the problem domain of the dataset
- the licensing conditions of the dataset (if any)
- the way in which the dataset was collected
description of the content of the dataset (including references to the information sources used) 	- number of data objects in the dataset
- the representation of the features (attributes) of the dataset together with their roles in the Orange tool
- the number of classes in the dataset, the meaning of each class and the way the classes are represented (explanation of the labels corresponding to the classes); if the dataset provides several possible classifications of the data, the report must identify which classification exactly is being considered in the work
- the number of data objects belonging to each class
- the number and meaning of the features (attributes) in the dataset, as well as their value types and ranges (this information must be represented in a table, indicating the feature title, explanation, value type and range of values available in the dataset) 
- a snippet of the data file structure showing all columns of the data file and their values for at least some data objects
conclusions drawn from the analysis of scatter plots, histograms and distributions (see Step 5 of Part I) on the separability of the classes in the dataset. Students must answer the given questions 	- Are the classes in the dataset balanced, or does one class (or several classes) prevail? It is determined by judging how many data objects belong to each class.
- Does the visual representation of the data allow the structure of the data to be seen? It is about whether data objects belonging to different classes can be clearly separated.
- How many data groupings can be identified by studying the visual representation of the data? It is about whether there are any separable groupings of data or data objects of different classes are merged.
- Are the identified data groupings close to each other or far apart?
conclusions from the analysis of statistical indicators (at least the central tendency and the dispersion of the feature values) 	 
Unsupervised machine learning

Goal: This part of the assignment aims to investigate the dataset further using clustering algorithms to see if the conclusions drawn earlier about the structure of the dataset and the separability of classes are valid.

The student team is required to do the following to complete this part of the assignment:

    Apply the two unsupervised machine learning algorithms considered in the course: (1) Hierarchical clustering and (2)  K-means algorithm.
    For the Hierarchical clustering algorithm, perform at least 3 experiments by freely moving the cut-off line and analysing how the number and content of the clusters change..
    Calculate the Silhouette coefficient for the K-means algorithm for at least 5 different values of k and analyse the performance of the algorithm.

Include the following information into the report:

    Description of the hyperparameters available in the Orange tool and their meaning for each of the algorithms (including references to the information sources used).
    Description of the experiments performed with each of the algorithms, clearly indicating the values of the hyperparameters used and providing conclusions on the performance of the algorithm in terms of how the results obtained correspond to the known number of classes in the dataset.
    Conclusions whether the classes in the dataset are well or poorly separable based on the analysis of the performance of the two algorithms.

Supervised machine learning

Goal: This part of the assignment aims to apply at least 3 classification algorithms to the previously analysed dataset and selected features of data objects.

One of the algorithms that must be used is artificial neural networks. The student team is free to choose two other algorithms.

The student team must do the following to complete this part of the assignment:

    Choose at least two supervised machine learning algorithms for the classification task. Students may use the algorithms covered in the study course or any other algorithms designed for the classification task. In the Orange tool these could be Logistic Regression, kNN, Tree, RandomForest, Gradient Boosting, SVM, Naive Bayes, AdaBoost.
    Split the dataset into training and test datasets.
    Perform at least 3 experiments with each algorithm using the training dataset, changing the values of the algorithm hyperparameters and analysing the performance metrics of the algorithms. 
    For each algorithm, select the trained model that provides the best algorithm performance.
    Apply the trained model for each algorithm to the test dataset.
    Evaluate and compare the performance of the trained models.

Include the following information into the report:

    A brief description (1/3 of an A4 page) of the two freely chosen algorithms and the rationale for their choice (excluding artificial neural networks), including references to the sources of information used.
    Description of the hyperparameters available in the Orange tool and their meaning for each of the algorithms.
    Information on the test and training datasets: (a) the total number of data objects added to the test and training datasets (number and %) and (b) information on how many data objects from each class are included in the training and test datasets (number and %).
    Hyperparameter values used in the experiments with each of the algorithms (in a table format) and screenshots showing these values and the performance metrics of the experiments.
    Conclusions on the performance of the models in the experiments performed, clearly identifying the model that will be used for testing.
    Results of the testing of the trained models and a comparison and interpretation of their performance, clearly separated from the training experiments. 

Maximum points:     15 points

    report – 3 points
    the developed project – 3 points
    defence – 5 points
    average number of points in peer evaluation – 4 points

The assignment will be automatically failed, and the defence will not be organised if:

    a breach of academic integrity has been identified;
    the student team has selected a dataset that is not suitable for the classification task and does not meet the requirements described;
    the student team has not provided a link to the project and dataset on the title page of the report..

