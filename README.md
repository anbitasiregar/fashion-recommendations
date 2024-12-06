# Fashion Trend Analysis & Machine Learning

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
June and July are the months with highest number of purchases, and October and December have the lowest number of purchases.

### Insight #2
Black is (by far!) the most popular color of clothing in the last two months.

### Insight #3
Average basket size differs significantly by age group, with customers below 20 averaging smaller basket sizes.

### Winning model
**kNN** with an accuracy of **76%%**

## Conclusions

### Recommendation #1:
Adjust budget for month or season-based campaigns to boost sales in months with lagging purchases like October and December. Further research could be needed to understand customer behaviors throughout the year.

### Recommendation #2:
Increase inventory for black clothes and accessories, as this is the most popular color by far. Designers should ensure that there are multiple options for black articles of clothing.

### Recommendation #3:
 Since the average basket size differs significantly by age group, personalized marketing campaigns can be designed for each demographic. For example, younger customers have smaller basket sizes, so they can be targeted with bundle offers or discounts to increase basket size.

## Next Steps
- Feature Selection & Engineering: We only used descriptive data of each article of clothing and didn't use any metadata on the customers that purchased the clothing. Cherry picking customer features and creating a more comprehensive dataset for machine learning may provide more accurate predictions.
- Hyperparameter Tuning: Our accuracy score is not very high and may need more hyperparameter tuning. With more time, we can do a selective GridSearchCV with this model to understand which hyperparameters would be even better to put into our model.

## For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/anbitasiregar/fashion-recommendations/tree/main/notebooks) or review the [presentation](https://github.com/anbitasiregar/fashion-recommendations/blob/main/presentations/Fashion%20Trend%20Predictor%20Analysis%20Presentation.pdf)

Original data source: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

## Repository Structure
```bash
├── notebooks
├── presentations
├── .gitignore
├── README.md
```