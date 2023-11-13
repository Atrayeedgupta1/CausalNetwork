# Inference with Bayesian Network
Problem statement 1:<br>
Given a dataset. Identify the anomalies, find the feature causing the anomalous behavior and predict the correct values for that feature. <br> 
Problem statement 2:<br>
There is an existing alarm tree. Simulate the same thing in a Bayesian network and find the root cause of a problem given certain evidence in particular nodes. <br>
Problem statement 3:<br>
Create a Causal Bayesian network and make interventional and counterfactual queries to infer important insights. <br>
<br><br>
Why Bayesian network models?<br>
First and foremost, we needed a concrete class of models on which to demonstrate and apply our ideas. Second, we wanted to use probability theory as our foundation. Our use of probability theory comes from necessity: most AI application domains involve uncertainty, with which we need to deal with explicitly and from the start in a principled way. Finally, we are interested in deriving cause-effect relationships from data. Even though many classes of models can be used to represent uncertain domains—like decision trees, artificial neural networks etc. only in the Bayesian network literature do we find claims of being able to represent and learn directed causal relationships.<br>
They can be used for a wide range of tasks including reasoning, anomaly detection, diagnostics, causal prediction.
<br><br>
Preliminaries<br>
Bayesian network: <br>
A Bayesian network is a graphical representation of a probability distribution over a set of variables. It consists of two parts: the directed acyclic graph and a set of probability distributions one for each node conditioned on the parent. <br>
Example of a discrete network<br>
![Screenshot 2023-11-10 140047](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/8c111d24-4a19-4c14-a69a-f9861f716c1a) <br>
A node can represent one variable or many variables and can be discrete, continuous and functional.<br>
Here A, B and C are the discrete variables having two states each and the arrows represent direct dependencies. 
The edges may or may not be causal. <br>
The conditional probabilities for each of the variable are given <br><br>
P(A)<br>
![Screenshot 2023-11-03 211455](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/4e98e7c5-586d-4a14-b38b-286a5c91fbfd) <br>
P(B|A,C) <br>
![Screenshot 2023-11-03 211538](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/6a6345a4-412f-4d7b-b3fb-841e4d612afc) <br>
P(C|A) <br>
![Screenshot 2023-11-03 211747](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/66b3092c-6564-43f3-8ed7-ea66b51bb484) <br>

The Bayesian network represent the joint probability distribution of the domain <br>
![Screenshot 2023-11-03 235548](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/78829185-0955-46ba-bdee-244b7746c7b9) <br>

Where Pa is the set containing the parents of X in the Bayesian network.
The probability distribution for A is the prior belief when nothing else is known. <br><br>
Distribution for C is dependent on A and is calculated using this <br>
P(C=State1) =P(C=State1|A=State1) P(A=State1) + P(C=State1|A=State2) P(A=State2) <br><br>
Similarly, distribution on B is dependent on A and C both and is calculated using this <br>
P(B=State1)= P(B=State1|A=State1,C=State1)P(A=State1)P(C=State1)+
 P(B=State1|A=State1,C=State2) P(A=State1)P(C=State2)+ 
 P(B=State1|A=State2,C=State1) P(A=State2)P(C=State1)+ 
 P(B=State1|A=State2,C=State2) P(A=State2)P(C=State2) <br><br>
Linear Gaussian BNs are based on continuous variables which are assumed to follow Gaussian distributions.
Hybrid BNs support both discrete and continuous distributions. It does not generally allow continuous variables 
to be parents of discrete ones. Sometimes it is useful to discretize continuous data, generating a discrete variable, 
where each state represents a continuous interval. <br><br>
When a link does not exist between two nodes, this does not mean that they are completely independent, as 
they may be connected via other nodes. They may however become dependent or independent depending on 
the evidence that is set on other nodes. Evidence is the information we know about a variable. If we are 100% 
certain about the information it is called the hard evidence otherwise it is called the soft evidence. <br><br>
During inferences or predictions in the Bayesian network it uses the Bayes theorem which states that <br><br>
![Screenshot 2023-11-03 233507](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/3a009dfb-7f5b-424f-b615-540eb8796ea5) <br><br>
The links and the conditional probabilities tables can be set manually using expert opinion. Otherwise, there are 
Structural learning algorithms for Bayesian networks, which can automatically determine the required links from 
data, this fall into two main classes. The first class is constraint-based methods that eliminate and orientate edges 
based on a series of conditional independence tests. The second class, score-based methods, represent a 
traditional machine learning approach where the aim is to search over different graphs maximising an objective 
function. The graph that maximises the objective function is returned as the preferred graph.<br><br>
Once the structure has been defined (i.e. nodes and links), a Bayesian network requires a probability distribution 
to be assigned to each node. Parameter learning is the process of using data to learn the distributions of a 
Bayesian network and it uses the maximum likelihood estimation procedure. <br><br><br> 
Problem statement 1<br><br>
Given the dataset. Identify the anomalies, find the feature causing the anomalous behavior and predict the 
correct values for that feature. <br><br>
The dataset was generated by a power plant and had continuous data though out. For the example purpose here, 
I take small sample from there with few features. It consists of 1450 observations without any missing value.<br>
Anomaly detection:<br>
Anomaly detection algorithms can be used in different health monitoring systems as well as in the preprocessing 
step to remove any anomalous point. Anomalous points are outliers or rare. Keeping them in our model make 
our inferences wrong. An observation becomes anomalous when one or more sensors give incorrect result. Here 
our goal is to predict the accurate value instead.<br>
I work here with a small train and test data and for verification purpose I introduce anomalies manually and do 
the same work I have done in the project.<br>
Bayesian networks are well suited for anomaly detection, because they can handle high dimensional data which 
humans find difficulties to interpret. While some anomalies are clearly visible by plotting individual variables, 
often anomalies are far more subtle, and are based on the interaction of many variables.<br>
The dataset<br><br>
![Screenshot 2023-11-03 195907](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/77ad846e-fa2a-44bd-ad43-24acfee00ff0) <br><br>
The model:<br><br>
Variables having same values throughout is ignored since they do not depict uncertainty in its occurring. Search 
and Score algorithm was used in Structural learning since it is sufficient to only consider correlation while doing 
the task. While doing parameter learning Relevance tree algorithm was used. Hence, we get the model like this 
where the arrows have no causal meaning. <br><br>
![Screenshot 2023-11-04 004930](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/7f4952b8-42f2-4f1a-8156-fca94070ef0f)<br><br>
We see how the variables are dependent on each other. We can condition on one variable and study the updated 
probability distributions on rest of the variables to get valuable insights. We used the normal procedure to build 
the model just because we know that the data with which we are creating the model is normal and does not 
contain any anomalous points. Else, we had to use some complex model like a Mixture model using Clustering 
algorithm to first find the anomalies using in-sample anomaly detection technique and then eliminate the 
anomalous points from the train data. After that we can build this model from normal points in the train dataset.<br>
Since we don’t have any anomalous point in the train data, we create this model using basic structural and 
parameter learning algorithms.<br><br>
Loglikelihood:<br>
If the result of learning is a model that does not contain information about the anomalous data, we have a model 
which represents normal behaviour. We can use this model to see how likely it is that unseen data could have 
been generated by this model. This tells us how anomalous the unseen data is. The lower the value, more 
anomalous the point is. The log-likelihood is simply the log of the probability density function (pdf) for the 
Bayesian network where the evidence from the test data is set.<br><br>
So, if this is our test data<br>
![Screenshot 2023-11-04 014226](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/78d2f001-4638-4012-8ec9-b264c4b9fe06)<br><br>
And we want to check which observations are anomalous we check the loglikelihood of each of the observations.<br>
![Screenshot 2023-11-13 134337](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/ed59c431-d584-4a3f-b835-f2f763e713b8) <br>
Looking at this we can conclude that points 1,3,4,5,9 are anomalies.<br><br>
Retracted Loglikelihood:<br>
When the anomaly score detects anomalous behaviour, we are usually then interested in diagnosing the cause 
of that anomaly. We can do this by testing the anomaly score without evidence set on a subset of particular 
variables in the Bayesian network. Below is the retracted loglikelihood for each of the feature for all of the 
observations in the test data.<br><br>
![Screenshot 2023-11-13 134558](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/21b7499a-fefc-4807-a67e-ed128ada02e2)
![Screenshot 2023-11-13 134613](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/473b5ac4-3c5f-4925-8a52-edc4b74fd661)<br><br>
Checking each observation at a time we conclude for each anomalous observation the variable giving the largest
negative retracted loglikelihood i.e., the variable giving the most impact for the anomalous behavior of the 
observation is<br><br>
Point 1: mainsteamtemp_esv <br>
Point 3: hpexhstpresr <br>
Point 4: load <br>
Point 5: hpexhstpresr <br>
Point 9: coolnwateroutlttemp <br>

Hence these are the most contributing factors for their anomalous behavior. If the retracted log likelihood is 
around 0 then no other factors are there which contributes to their anomalous behavior otherwise there are 
more factors.<br><br>
For point 1, removing the effect of mainsteamtemp_esv we get mainsteampresr_esv having largest negative 
retracted loglikelihood<br>
![Screenshot 2023-11-13 134628](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/96253d29-f3b3-4e87-9c5a-311b60be564a)<br>
Removing the effect of mainsteamtemp_esv, mainsteampresr_esv we get hpexhsttemp having largest negative 
retracted loglikelihood<br>
![Screenshot 2023-11-13 134645](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/336e4357-fa8a-4677-a973-ee3aefca43cb)<br>
For Point 3, removing the effect of hpexhstpresr we get msflow having the largest negative retracted 
loglikelihood,
![Screenshot 2023-11-13 134701](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/c6e73a53-9d56-4889-bc7e-e0d3367ddb22)<br>
For Point 4, removing the effect of load we get coolnwateroutlttemp having the largest negative retracted 
loglikelihood,
![Screenshot 2023-11-13 134739](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/215130a0-57a4-4ca6-aa75-83c1dbc07d30)<br>
Removing the effect load, coolnwateroutlttemp we get mainsteampresr_esv having the largest negative 
retracted loglikelihood
![Screenshot 2023-11-13 134756](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/1878538e-e3fe-4e3e-b8ec-da48b7e6638b)<br>
Removing the effect load, coolnwateroutlttemp, mainsteampresr_esv we get msflow having the largest 
negative retracted loglikelihood
![Screenshot 2023-11-13 134808](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/56253ab1-aef0-45a7-a115-63aa7d731455)<br>
For Point 5, removing the effect of Hpexhstpresr we get mainsteamtemp_esv having the largest negative 
retracted loglikelihood
![Screenshot 2023-11-13 134854](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/98dae84e-96f3-480a-b31c-92d3b57e104b)<br>
For Point 9, removing the effect of coolnwateroutlttemp we get mainstreamtemp_esv 
![Screenshot 2023-11-13 134909](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/d148358b-4e51-4bc7-b2b2-98d755395333)1<br>
Removing the effect of coolnwateroutlttemp ,mainstreamtemp_esv we get msflow
![Screenshot 2023-11-13 134919](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/fde3495f-90d8-4920-b1f7-85a1ad358980)<br><br>
Therefore, the traced factors for all the observations in the test dataset are<br>
![Screenshot 2023-11-13 134939](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/6dbec6fc-8f94-4cc0-b512-791417f7e12c) <br>
Thus, we are successful in diagnosing the root cause of the problem as well. Our next aim is to predict the correct 
values for these places.<br><br>
Prediction:<br>
We use batch query method because it allows multiple cases to be queried at once. Prediction is the process of 
calculating a probability distribution over one or more variables whose values we would like to know, given 
information (evidence) we have about some other variables. The variables we are predicting are known 
as Output variables, while the variables whose information we are using to make the predictions are known 
as Input variables. In a Bayesian network any variable can be treated as an output, also any variable can be 
treated as an input. <br>
Hence after prediction for each of the observations we get <br><br>
![Screenshot 2023-11-13 134954](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/0c3183dd-bd87-464a-b2f9-dee31732d4de) <br>
When a Bayesian network has been built from data, it is common practice to evaluate the performance. Since 
the variables are continuous, I used the metric R squared to understand how well the model is performing. R squared, also known as the Coefficient of determination is a standard metric which tells us how well 
the inputs explain the variance of the output. Its value is between 0 and 1(the closer to 1 is better). For our final 
model we got a good R-squared value.




























 
