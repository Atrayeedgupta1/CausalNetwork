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
loglikelihood,<br>
![Screenshot 2023-11-13 134701](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/c6e73a53-9d56-4889-bc7e-e0d3367ddb22)<br>
For Point 4, removing the effect of load we get coolnwateroutlttemp having the largest negative retracted 
loglikelihood,<br>
![Screenshot 2023-11-13 134739](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/215130a0-57a4-4ca6-aa75-83c1dbc07d30)<br>
Removing the effect load, coolnwateroutlttemp we get mainsteampresr_esv having the largest negative 
retracted loglikelihood<br>
![Screenshot 2023-11-13 134756](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/1878538e-e3fe-4e3e-b8ec-da48b7e6638b)<br>
Removing the effect load, coolnwateroutlttemp, mainsteampresr_esv we get msflow having the largest 
negative retracted loglikelihood<br>
![Screenshot 2023-11-13 134808](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/56253ab1-aef0-45a7-a115-63aa7d731455)<br>
For Point 5, removing the effect of Hpexhstpresr we get mainsteamtemp_esv having the largest negative 
retracted loglikelihood<br>
![Screenshot 2023-11-13 134854](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/98dae84e-96f3-480a-b31c-92d3b57e104b)<br>
For Point 9, removing the effect of coolnwateroutlttemp we get mainstreamtemp_esv <br>
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
model we got a good R-squared value.<br><br><br>
Problem statement 2<br><br>
There is an existing alarm tree. Simulate the same thing in a Bayesian network and find the root cause of a 
problem given certain evidence in particular nodes.<br>
Fault detection and their diagnosis play an essential role in the industry. The search for signatures or 
fault indicators has as a purpose to characterize the operation of the system by identifying the type 
and origin of each of the failures. Various approaches developed for this purpose can be mainly divided 
into two categories. The first is mathematical model based, such as multinomial logistic regression and 
Bayesian networks. A fault tree is considered to simplify determining causality between components. 
Any fault tree can be transformed into a corresponding Bayesian network by creating a binary Bayesian 
network node for each event in the fault tree. The method of fault tree is widely used in the field of 
the reliability. It offers a framework privileged to the deductive and inductive analysis by means of a 
tree structure of logical gates. The procedure that uses fault trees for diagnosis purposes is abductive, 
focusing first on adverse events and then identifying their causes. A fault tree is established as a logical 
diagram and has the undesirable event at the top. The immediate causes that produce this event are 
then hierarchized using logical symbols "AND" and "OR". To perform a correct diagnosis from the fault 
trees, these must largely represent all the causal relationships of the system, capable of explaining all 
possible fault scenarios.<br>
The advantage of probabilistic graphical models is interesting graphical representation of models, easy 
to understand and analyse. In addition, the probabilistic failure analysis evaluates the probability of 
failure of a complex system that its weak points can be identified.<br>
The tree has conditions and the final outcome or the leaf node. These leaf nodes describe what thing we have 
to check when we get faulty results as inputs. For example, as the leaf nodes we have ‘CEP ineffective OR suction 
trainer choked’, ‘CEP A motor overload protection’ etc.<br>
We have two existing alarm trees one for Hotwell level high another one for Hotwell level low. We have to 
combine them both to form a Bayesian network. It is converted to a Bayesian network so that we can do more 
flexible and clean queries.<br><br>
![Screenshot 2023-11-13 143814](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/b50d2295-75b4-4070-8d2b-6299cadabe85)<br><br>
![Screenshot 2023-11-13 143823](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/84b28056-1313-4eb2-8069-b2f4eb36612f)<br><br>
Building bayesian network from the fault tree is to transform the graphical representation of the fault 
tree into bayesian network. Events and logic Gates (AND, OR) are the basic elements for the fault 
tree. However, the bayesian network use as basic elements nodes that representing events and arcs 
that model the dependences between events and relations causes – effect. <br><br>
The final Bayesian network structure combining both the trees looks like this, here we have to specify the 
probability distributions.<br>
![Screenshot 2023-11-13 143845](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/5563f61b-2504-4eed-8a10-092160f253f3)br><br>
The leaf nodes are the root causes and after setting the evidence we get out conclusions about the root nodes. 
Hence like this we are able to diagnose the system.<br>
Let’s check on one network to see how it functions<br>
![Screenshot 2023-11-13 143903](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/a9f5aa27-5a2d-40f5-884d-a5f965173ebb)<br><br>
The causal links are derived from the alarm tree and the conditional distribution tables acts as an OR gate. The 
distribution on the leaf nodes is specified by the experts<br><br>
Let’s say we have evidence about the given nodes<br>
![Screenshot 2023-11-13 143918](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/6971eb17-987d-449b-ae69-e34d6ac2445f)<br>
<br>
It is clear from the model that the root cause is CEPbMtrOverloaded.<br>
When we don’t have enough evidence and we cannot be very certain about a particular root cause we use the 
Value of Information technique which helps us to know which other variables if known can reduce the 
uncertainty of the root cause.<br><br><br>
Problem statement 3<br>
Create a Causal Bayesian network and make interventional and counterfactual queries to infer important 
insights. <br><br>
Correlation and causation:<br>
Correlation/ association is the statistical dependency between two variables. Two types are positive correlation 
and negative correlation and this is denoted by the Pearson’s correlation coefficient ‘r’. All of the machine 
learning models that we use takes this into consideration, but it does not make the machine intelligent like 
humans. <br>
E.g. – Sleeping with our shoes has positive correlation with having head ache the next day, but this has no 
meaning we cannot say that sleeping with shoes is a cause of the head ache.<br><br>
Causation is when one affects the other. B happens only if A happens.<br>
E.g. – I take medicine and I become healthy. Had I not taken the medicine I would have stayed sick. Thus, medicine 
is the cause and healthy is the effect.<br><br>
Confounder:<br>
Confounder is the common cause of treatment and effect. In the first example that we considered drinking is the 
common cause of both sleeping with shoes and having head ache the next day. This is the reason why we have 
positive correlation between the two variables.<br>
Thus, correlation is the mixture of confounding association and causal association. When we want to study about 
the causal nature of the problem e have to remove the confounders. This can be done using several techniques 
like the backdoor adjustment, front door adjustment etc. <br><br>
Causal model:<br>
This is the Bayesian model where all the edges represent cause and effect relationship.<br><br>
Individual /Average treatment effect:<br>
Potential outcomes are called potential because they didn’t actually happen. Instead, they denote what would 
have happened in the case some treatment was taken.<br><br>
Individual treatment effect = Y(A=1) – Y(A=0)<br><br>
Of course, due to the fundamental problem of causal inference, we can never know the individual treatment 
effect because we only observe one of the potential outcomes. For the time being, let’s focus on something 
easier than estimating the individual treatment effect. Instead, lets focus on the average treatment effect, which 
is defined as follows.<br><br>
ATE = E [Y1 -Y0] <br><br>
Where Y1 is the output when the treatent is 1 and Y0 is the output when the treatment is 0.<br>
Another easier quantity to estimate is the average treatment effect on the treated<br>
ATT = E [Y1-Y0 | T=1] <br><br>
Association is measured by E [Y|T=1] – E [Y|T=0]<br><br>
Causation is measured by E [ Y1-Y0 ]<br>
Where<br>
E [Y|T=1] – E [Y|T=0] = E [Y1-Y0 | T=1] + { E [Y0|T=1] – E [Y0|T=0] }<br><br>
Here the first term is the average treatment effect given treated and the second term represents the bias.<br>
I.e., Bias = Affected not given treatment – unaffected not given treatment<br>
The bias is given by how the treated and the control group differ before the treatment, in case neither of them 
has received the treatment.<br>
Two groups are comparable only if their effects are same when no treatment is given<br><br>
Randomized Control Trials:<br>
Now, we look at the first tool we have to make the bias vanish by randomised experiments. Randomised 
experiments randomly assign individuals in a population to a treatment or to a control group. The proportion 
that receives the treatment doesn’t have to be 50%. You could have an experiment where only 10% of your 
samples get the treatment.<br><br>
Intervention:<br>
Overriding a variable is different to observing its behaviour. When we override, we call it an intervention. 
Conditioning on T=t just means that we are restricting our focus to the subset of the population to those who 
received treatment t. In contrast, an intervention would be to take the whole population and give everyone 
treatment t.<br><br>![Screenshot 2023-11-14 153430](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/941e2ac9-4d32-432b-8ccf-8eacfc163030)<br><br>
We denote intervention with the do operator do(T=t)<br><br>
Conditional independence and d-separator:<br>
In a chain junction, A -> B -> C controlling for B prevents information about A from getting to C or vice versa. <br>
In a fork, A <- B -> C controlling for B prevents information about A from getting to C or vice versa.<br>
In a collider, A ->B<-C exactly the opposite holds here. The variables start out independent but if you control for 
B then information starts flowing.<br>
Conditioning on descendants of a collider also induces association in between the parents of the collider. So,
conditioning on the descendant is similar to conditioning on the collider itself. <br>
So whenever two variables are conditionally independent that means that the path between them is ‘blocked’<br>
The flow of association is symmetric, whereas the flow of causation is not. Causation only flows in a single 
direction<br>
Two set of nodes are called d-separated by a set of nodes Z if all the paths between X and Y are blocked by Z.<br><br>
Backdoor criterion:<br>
This is a method where we control for confounders to find the actual causal effect.<br>
The nondirected unblocked paths from T to Y are known as backdoor paths. And it turns out that if we can block 
these paths by adjusting (i.e., making it a constant value), we can identify causal quantities. I.e., close all backdoor 
paths while leaving the front door paths.<br>

A set of variables {Z} satisfies the backdoor criterion relative to an ordered pair of variables (Treatment(T) and 
Outcome(Y) in a DAG if:<br>
1. No node in {Z} is a descendant of T.<br>
2. {Z} blocks every path between T and Y that contain an arrow into T (called the backdoor path)<br><br>
Hence the causal effect of T on Y is <br>
P(y |do(t)) = ∑P(y |t, w)P(w)<br><br>
![Screenshot 2023-11-14 153440](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/eb06d91a-d52d-4385-978b-1b788f79488d)<br>
If we don’t have data for the conditioning variables then this strategy won’t work.<br><br>
Front door adjustment:<br>
When we cannot collect data on the confounders, we cannot use the backdoor criterion. Instead, we use the 
front door adjustment method. The steps to perform front door adjustment are<br>
1. Identify the causal effect of T on M.<br>
2. Identify the causal effect of M to Y<br>
3. Combine the above steps to identify the causal effect of T on Y.<br><br>
![Screenshot 2023-11-14 153451](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/32e5c93e-519e-4838-8635-517a89c0fb11)<br><br>
Disjunctive cause criterion:<br>
When we don’t have a causal model to make causal queries, we use this method.<br>
Here we control for each covariate that is a cause of the exposure, or of the outcome, or both; exclude from this 
set any variable known to be an instrumental variable and include as a covariate any proxy for an unobserved 
variable that is a common cause of both the exposure and the outcome. <br><br>
![Screenshot 2023-11-14 153501](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/fcd5d62f-bd7c-45d9-8093-eeda778fdf1c)<br><br>
Building up a causal model:<br>
There are some of the algorithms available which helps us to get graphs up to the Markov equivalent class. Here 
we use PC algorithm which is a constraint-based method. To get accurate results with this algorithm we need a 
large dataset.<br><br>
Assumptions to make are<br>
1. Faithfulness assumption<br>
A graph is said to be faithful if the conditional independencies found in the data are reflected in the graph. i.e.,
Two variables are independent implies they are d-separated.<br>
2. Causal sufficiency <br>
There are no unobserved confounders. i.e., no two variables share an unobserved common cause.<br><br>
Markov equivalence class:<br>
Two graphs are called Markov equivalent if and only if they have the same colliders and same skeleton.<br>
PC algorithm helps us to find the Markov equivalence class using conditional independence tests. The most 
common CI test for discrete variables is the chi squared test and for continuous variable it is the fisher’s z test.<br>
Thus using the data we have, we use PC algorithm to get a structure like this.<br><br>
![Screenshot 2023-11-14 153511](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/5fc0b11c-b3f0-40ad-9bcc-143205adae2e)<br><br>
We further verify the network by expert opinion and by parameter learning we learn the corresponding 
conditional probability tables. We can make causal queries using this model .A counterfactual is a hypothetical 
or "what-if have happened if things would be different”.<br>
Do-calculus offers no way of connecting the information across the different worlds thus we have to make use 
of the characteristic variable and not just depend on the interventions<br>
![Screenshot 2023-11-14 153527](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/3eb976e9-d77f-4528-9ae1-672afbda3855)<br><br>
Consider an example to understand the counterfactual queries,<br>
Suppose we have<br>
![Screenshot 2023-11-14 161417](https://github.com/Atrayeedgupta1/CausalNetwork/assets/109009826/4aeb218c-9531-4bd7-883a-766d28376fdc)<br><br>
We want to know under the same situation what would be my load if msflow was 500.<br><br>
There are three steps to perform such query<br>
1. Abduction – use the data about caseid 1 to estimate its idiosyncratic factors U1, U11, U12, U13, U14, 
U15, U16 for this case.<br>
2. Action - Use the do-operator to change the model to reflect the counterfactual assumption being 
made, in this case that it has msflow 500<br>
3. Prediction – Calculate this caseid’s new load using the modified model and the updated information 
about the exogenous variables U1, U11, U12, U13, U14, U15, U16.<br><br>
Like this we can get a lot of more information just by making queries from this causal network.<br><br><br>
Conclusion<br><br>
Thus, we emphasized the importance of Bayesian networks in various used cases. The incorporation 
of prior knowledge and the dynamic adjustment of beliefs as new information surfaces equip 
practitioners with a robust methodology for navigating uncertainty and making well informed 
decisions.<br>
As we reflect on the journey through the intricacies of Bayesian networks, it is clear that the 
significance of this framework extends beyond a mere statistical tool. It is a gateway to unveiling 
hidden patterns, understanding complex systems, and harnessing the power of probabilistic
reasoning for enhanced decision making.






























































 
