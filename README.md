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
where the arrows have no causal meaning.






 
