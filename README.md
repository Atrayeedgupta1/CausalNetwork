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



 
