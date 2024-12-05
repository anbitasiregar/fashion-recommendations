# Fashion Trend Predictor

## Objective
This [dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) comes from Kaggle and details purchase history from the popular clothing store H&M from September 20, 2018 to September 22, 2020. We will use this data to:
1. analyze purchase and clothing trends.
2. provide outfit recommendations to repeat customers of H&M.

## Business Problem
This analysis is great for H&M and other retailers to enhance customer engagement and drive sales leverage purchase data.
- Personalized Recommendations: Retailers can provide outfit suggestions based on a fashion trends and purchase history to increase cross-selling opportunities.
- Inventory Management: Identify short and long-term trends to optimize stock -- reducing overstock and anticipating customer demand for popular products.
- Customer Retention: Insights from purchase behavior can be used to target customers with campaigns and deals on trendy items.

## Data Understanding
There are 3 datasets we will explore. `articles.csv` has descriptive data on different articles of clothing sold at H&M, `customers.csv` has metadata on H&M customers, and `transactions_train.csv` is our training data for our model and contains data on each transaction within the two-year timeframe described above.

Target feature: `popularity`
NOTE: the target feature was calculated from number of purchases per article item.

## Analysis & Results

### Insight #1

### Insight #2

### Insight #3

### Winning model
**Random Tree Classifier** with an accuracy of **74%%**

## Conclusions

### Recommendation #1:

### Recommendation #2:

### Recommendation #3:

## Next Steps

## For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/anbitasiregar/fashion-recommendations/tree/main/notebooks) or review the [presentation](https://github.com/anbitasiregar/nasa-asteroid-analysis/blob/main/presentations/Asteroid%20Classification%20Analysis%20Presentation.pdf)

Original data source: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

## Repository Structure
```bash
├── notebooks
├── presentations
├── .gitignore
├── README.md
```