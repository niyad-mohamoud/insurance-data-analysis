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

**Key Outcomes:**  
- Identified trends in claim likelihood by age and gender  
- Observed how claim value relates to age and distance from London  
- Noted outliers and anomalies in the dataset for further insight  

**How to Run:**  
1. Clone the repository.  
2. Install required libraries: `pip install pandas numpy matplotlib seaborn scikit-learn openpyxl`  
3. Run the Python scripts in `scripts/` folder to reproduce the analysis.
