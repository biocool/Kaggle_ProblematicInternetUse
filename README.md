# Child Mind Institute: Problematic Internet Use Prediction

This project is part of a [Kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) aimed at predicting problematic internet use among children and adolescents using physical activity and fitness data. The model we develop will help identify early indicators of excessive internet use, enabling timely interventions to promote healthier digital habits. The competition uses data from the **Healthy Brain Network**, linking physical changes to mental health challenges like depression and anxiety.

### Goal
The objective is to build a predictive model that analyzes fitness data to identify problematic internet usage patterns, particularly in contexts where clinical expertise is limited.

### Acknowledgments
This project utilizes data from the Healthy Brain Network, with support from Kaggle, Dell Technologies, and NVIDIA. 

### Team Contributions
We aim to develop a robust model by leveraging machine learning techniques to build an AI-driven mental health diagnostic tool.

### Example with \(N=4\)

1. **Actual and Predicted Labels**  
   Actual Labels: [0, 1, 2, 2, 3, 1, 0, 3, 2, 1]  
   Predicted Labels: [0, 2, 1, 3, 3, 1, 0, 2, 2, 1]

2. **Actual Histogram**  
   - Count of 0: 2  
   - Count of 1: 3  
   - Count of 2: 3  
   - Count of 3: 2  
   Actual Histogram: \([2, 3, 3, 2]\)

3. **Predicted Histogram**  
   - Count of 0: 2  
   - Count of 1: 3  
   - Count of 2: 3  
   - Count of 3: 2  
   Predicted Histogram: \([2, 3, 3, 2]\)

4. **Weight Matrix \(W\)**  
   Using the formula:  
   \[
   W_{i,j} = \frac{(i-j)^2}{(N-1)^2}
   \]  
   Weight Matrix \(W\):  
   \[
   W = \begin{bmatrix}
   0 & 0.11 & 0.44 & 1 \\
   0.11 & 0 & 0.11 & 0.44 \\
   0.44 & 0.11 & 0 & 0.11 \\
   1 & 0.44 & 0.11 & 0
   \end{bmatrix}
   \]

5. **Calculation of Quadratic Weighted Kappa (\(\kappa\))**  
   Observed Matrix \(O\):  
   \[
   O = \begin{bmatrix}
   2 & 0 & 0 & 0 \\
   0 & 0 & 1 & 2 \\
   0 & 1 & 2 & 0 \\
   0 & 2 & 0 & 2
   \end{bmatrix}
   \]  
   Expected Matrix \(E\):  
   \[
   E = \frac{1}{10} \begin{bmatrix}
   2 & 3 & 3 & 2 \\
   2 & 3 & 3 & 2 \\
   2 & 3 & 3 & 2 \\
   2 & 3 & 3 & 2
   \end{bmatrix} = \begin{bmatrix}
   0.4 & 0.6 & 0.6 & 0.4 \\
   0.4 & 0.6 & 0.6 & 0.4 \\
   0.4 & 0.6 & 0.6 & 0.4 \\
   0.4 & 0.6 & 0.6 & 0.4
   \end{bmatrix}
   \]  

   Calculation:  
   \[
   \kappa = 1 - \frac{\sum W O}{\sum W E}
   \]  
   Assuming:  
   \(\sum W O = 3.8\)  
   \(\sum W E = 5.0\)  
   Substituting these values:  
   \[
   \kappa = 1 - \frac{3.8}{5.0} = 1 - 0.76 = 0.24
   \]  

   **Final Score**  
   Quadratic Weighted Kappa: \(0.24\)



### License
[MIT License](LICENSE)
