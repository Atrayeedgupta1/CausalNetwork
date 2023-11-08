# CausalNetwork
Leveraged Bayesian networks to develop a sophisticated anomaly detection system. This involved detecting anomalous data points and subsequently conducting in-depth diagnostics to identify underlying factors and then predicting the correct values. Integrated causality within the network, enabling interventional and counterfactual simulations.
Inference with Bayesian Network and Causality

Problem statement 1:
Given a dataset. Identify the anomalies, find the feature causing the anomalous behavior and predict the correct values for that feature. 
Problem statement 2:
There is an existing alarm tree. Simulate the same thing in a Bayesian network and find the root cause of a problem given certain evidence in particular nodes. 
Problem statement 3:
Create a Causal Bayesian network and make interventional and counterfactual queries to infer important insights. 

Why Bayesian network models?
First and foremost, we needed a concrete class of models on which to demonstrate and apply our ideas. Second, we wanted to use probability theory as our foundation. Our use of probability theory comes from necessity: most AI application domains involve uncertainty, with which we need to deal with explicitly and from the start in a principled way. Finally, we are interested in deriving cause-effect relationships from data. Even though many classes of models can be used to represent uncertain domains—like decision trees, artificial neural networks etc. only in the Bayesian network literature do we find claims of being able to represent and learn directed causal relationships.
They can be used for a wide range of tasks including reasoning, anomaly detection, diagnostics, causal prediction.

Preliminaries
Bayesian network: 
A Bayesian network is a graphical representation of a probability distribution over a set of variables. It consists of two parts: the directed acyclic graph and a set of probability distributions one for each node conditioned on the parent. 
Example of a discrete network
 
