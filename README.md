## Car Insurance Data Analysis Project

This project analyses a car insurance dataset to answer three key questions using Python and data science tools:

1. **Does the chance of making a claim depend on gender or age?**  
   - Calculated claim probabilities for men and women.  
   - Grouped customers into age bands (5-year and 10-year intervals) to visualize claim likelihood.  

2. **Does the value of a claim tend to increase with age?**  
   - Built a Decision Tree Regressor to predict claim values based on customer age.  
   - Compared predicted claim values to actual values and plotted results to observe patterns and outliers.  

3. **Does the value of a claim tend to increase nearer to London?**  
   - Converted postcodes to latitude and longitude.  
   - Calculated distances to London using the Haversine formula.  
   - Applied a Decision Tree Regressor to analyze the relationship between distance from London and claim value.  

**Tools & Libraries Used:**  
- Python: pandas, numpy, openpyxl  
- Visualization: matplotlib, seaborn  
- Machine Learning: scikit-learn (DecisionTreeRegressor)  


**Project Structure:**
- `Q1.py` — Analyses claim probabilities by gender and age group.
- `Q2.py` — Uses decision tree regression to explore claim value in relation to customer age.
- `Q3.py` — Calculates geographical distances from London using postcode coordinates and the Haversine formula, then analyses claim value against distance.
- `README.md` — Project overview, methodology and instructions.

**Key Outcomes:**
- Analysed claim frequency across gender groups and age bands.
- Explored the relationship between customer age and claim value using decision tree regression.
- Investigated geographical distance from London in relation to claim value.
- Evaluated model predictions using Mean Absolute Error (MAE) and visualised actual values, predicted values and potential outliers.
nsight  

**How to Run:**  
1. Clone the repository.  
2. Install required libraries: `pip install pandas numpy matplotlib seaborn scikit-learn openpyxl`  
3. Run Q1.py, Q2.py and Q3.py from the repository's root directory.
