# Child Mind Institute: Problematic Internet Use Prediction

This project is part of a [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) aimed at predicting problematic internet use among children and adolescents using physical activity and fitness data. The model we develop will help identify early indicators of excessive internet use, enabling timely interventions to promote healthier digital habits. The competition uses data from the **Healthy Brain Network**, linking physical changes to mental health challenges like depression and anxiety.

### Goal
The objective is to build a predictive model that analyzes fitness data to identify problematic internet usage patterns, particularly in contexts where clinical expertise is limited.

### Acknowledgments
This project utilizes data from the Healthy Brain Network, with support from Kaggle, Dell Technologies, and NVIDIA. 

### Team Contributions
We aim to develop a robust model by leveraging machine learning techniques to build an AI-driven mental health diagnostic tool.

Example with 
𝑁
=
4
N=4
1. Actual and Predicted Labels
Actual Labels: [0, 1, 2, 2, 3, 1, 0, 3, 2, 1]
Predicted Labels: [0, 2, 1, 3, 3, 1, 0, 2, 2, 1]
2. Actual Histogram
Count of labels:

Count of 0: 2
Count of 1: 3
Count of 2: 3
Count of 3: 2
Actual Histogram: 
[
2
,
3
,
3
,
2
]
[2,3,3,2]

3. Predicted Histogram
Count of labels:

Count of 0: 2
Count of 1: 3
Count of 2: 3
Count of 3: 2
Predicted Histogram: 
[
2
,
3
,
3
,
2
]
[2,3,3,2]

4. Weight Matrix 
𝑊
W
Using the formula:

𝑊
𝑖
,
𝑗
=
(
𝑖
−
𝑗
)
2
(
𝑁
−
1
)
2
W 
i,j
​
 = 
(N−1) 
2
 
(i−j) 
2
 
​
 
Weight Matrix 
𝑊
W:

𝑊
=
[
0
0.11
0.44
1
0.11
0
0.11
0.44
0.44
0.11
0
0.11
1
0.44
0.11
0
]
W= 
​
  
0
0.11
0.44
1
​
  
0.11
0
0.11
0.44
​
  
0.44
0.11
0
0.11
​
  
1
0.44
0.11
0
​
  
​
 
5. Calculation of Quadratic Weighted Kappa (
𝜅
κ)
To compute 
𝜅
κ:

Observed Matrix 
𝑂
O (frequency of actual vs. predicted):
𝑂
=
[
2
0
0
0
0
0
1
2
0
1
2
0
0
0
0
2
]
O= 
​
  
2
0
0
0
​
  
0
0
1
0
​
  
0
1
2
0
​
  
0
2
0
2
​
  
​
 
Expected Matrix 
𝐸
E (using outer product of histograms):
𝐸
=
1
10
[
2
3
3
2
2
3
3
2
2
3
3
2
2
3
3
2
]
=
[
0.4
0.6
0.6
0.4
0.4
0.6
0.6
0.4
0.4
0.6
0.6
0.4
0.4
0.6
0.6
0.4
]
E= 
10
1
​
  
​
  
2
2
2
2
​
  
3
3
3
3
​
  
3
3
3
3
​
  
2
2
2
2
​
  
​
 = 
​
  
0.4
0.4
0.4
0.4
​
  
0.6
0.6
0.6
0.6
​
  
0.6
0.6
0.6
0.6
​
  
0.4
0.4
0.4
0.4
​
  
​
 
Calculation:
𝜅
=
1
−
∑
𝑖
,
𝑗
𝑊
𝑖
,
𝑗
𝑂
𝑖
,
𝑗
∑
𝑖
,
𝑗
𝑊
𝑖
,
𝑗
𝐸
𝑖
,
𝑗
κ=1− 
∑ 
i,j
​
 W 
i,j
​
 E 
i,j
​
 
∑ 
i,j
​
 W 
i,j
​
 O 
i,j
​
 
​
 
Assuming the calculations yield the following sums:

∑
𝑊
𝑂
=
3.8
∑WO=3.8
∑
𝑊
𝐸
=
5.0
∑WE=5.0
Substituting these values:

𝜅
=
1
−
3.8
5.0
=
1
−
0.76
=
0.24
κ=1− 
5.0
3.8
​
 =1−0.76=0.24
Final Score
Quadratic Weighted Kappa: 
0.24
0.24


### License
[MIT License](LICENSE)
