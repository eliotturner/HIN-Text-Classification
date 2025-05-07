<h1>HIN Classification using R-GCN with Metadata</h1>
<h2>Project Summary</h2>
<p>Our project aims to devlop a classifier which can use an HIN with and without metadata to classify emails into their respective newsgroup topics. This serves as a good example of text-classification accuracy with difficult categorization tasks. We hope to discover the effects of metadata in the classificiation of text, and see if there are relationships that can improve accuracy which are not found in the body text alone.</p>
<h2>Necessary Downloads:</h2>
<p>
For this program to run, there are  a few larger dependencies that cannot be included in this repo 
but are necessary for the running of the code. This begins with our dataset the 20_newsgroups dataset. This dataset can be downloaded 
from the link below (download the file named: 20news-19997.tar.gz). The data must be extracted, and the folder called "20_newsgroups" must be placed in the root directory of the project.</p>
<p style="text-align:center"> <a href="http://qwone.com/~jason/20Newsgroups/">20-Newsgroups-Download</a> </p>
<p>The second dependency that must be downloaded are GloVe embeddings. This embedding can be dowloaded from the following source, and the file must be placed again in the source directory.</p>
<p style="text-align:center"> <a href="https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt">GloVe-Download</a> </p>
<h2>Code Summary</h2>
<h3>Utils.py</h3>
This file handles some helper function to clean up the train_rgcn file. Apart from this, its primary function is to parse through the 20_newsgroups, extract the necessary information, and create a DGL Heterograph HIN using the data for usage in training. It also provides an unused function which can display the HIN visually using Graphistry.
<h3>train_rgcn.py</h3>
<p>This file contains the model definition, and the training loop. This model instantiates the HIN using functions from Utils.py. The model, a Relational Graph Convolutional Network, is trained using the HIN and results are tracked throughout the training loop. A accuracy curve is produced as a .png, and the metrics are produced in the form of a metrics.csv and confusion_matrix.csv.</p>
<h3>dataprep.py</h3>
<p>This file simply calcluates class-wise, macro, and micro precision, recall, and F-1 scores. These are printed to the console.</p>
