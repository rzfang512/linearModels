# 1.

## 1.
# A model is linear if it is linear in the parameters.

## 2.
# The coefficient of each dummy is the group mean of the outcome for that category if without intercept. With intercept, the coefficient of the dummy is the difference
# between group and the baseline group.

## 3. No, because predictions are continuous, but class labels are discrete.

## 4. The signs when the linear model is overfitting when there is high R^2 on the training set, but low R^2 or high RMSE on the test set. When the model has too many variables,
# it would also create overfitting.

## 5. Multicollinearity means that two or more independent variables are highly correlated with each other. 2SLS helps when variable is endogenous and often high collinear with each others.
# By using 2SLS, we fix endogeneity and multicollinearity by filtering out the contaminated parts of the predictors.

## 6. We can incorporate nonlinear relationships between dependent variable y and independent variable by transforming variables or by using more flexible models.

## 7. Intercept is the predicted value of the dependent variable when all predictors or independent variable are 0. Slope coefficient is the change in y for a 1 unit increase in x.
# Coefficient for a dummy or one hot variable has two aspects. Without intercept, it is the mean value of y for that group. With intercept, the difference in mean y between that group and the
# reference group.

# 2.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

## 1.
df = pd.read_csv('Q1_clean.csv')

# Quick check
print(df.head())
print(df.columns)
print(df.describe())

## 2.
df.rename(columns={"Neighbourhood ": "Neighbourhood"}, inplace=True)

neighbourhood_stats = df.groupby("Neighbourhood")[["Price", "Review Scores Rating"]].mean()

neighbourhood_stats_sorted = neighbourhood_stats.sort_values(by="Price", ascending=False)
print(neighbourhood_stats_sorted)

# The coefficient from no intercept regression match the group mean prices exactly. When including an intercept, the intercept captures
# the baseline mean, and the coefficients measure how much more or less expensive each borough is compared to baseline.

## 3.
dummies_with_drop = pd.get_dummies(df["Neighbourhood"], drop_first=True)

X = dummies_with_drop
y = df["Price"]

model_with_intercept = LinearRegression()
model_with_intercept.fit(X, y)

intercept = model_with_intercept.intercept_
coef_with_intercept = pd.Series(model_with_intercept.coef_, index=X.columns)

print("Intercept:", intercept)
print("\nCoefficients:\n", coef_with_intercept)

# I handled by dropping the dummy varibale from part 2. The intercept becomes the average Price for the borough that was dropped.
# Each coefficient shows the difference in average Price between that borough and the reference borough.
# By adding the intercept to each coefficient to recover the original group means to get the original coefficients from part 2.

## 4.
train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = pd.get_dummies(train[["Review Scores Rating", "Neighbourhood"]], drop_first=True)
X_test = pd.get_dummies(test[["Review Scores Rating", "Neighbourhood"]], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train["Price"]
y_test = test["Price"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2_test = r2_score(y_test, y_pred)


rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))


r2_test, rmse_test

coef_review_score = pd.Series(model.coef_, index=X_train.columns)["Review Scores Rating"]
coef_review_score
neighbourhood_coefs = pd.Series(model.coef_, index=X_train.columns)
neighbourhood_coefs_neigh = neighbourhood_coefs.drop("Review Scores Rating")

most_expensive_borough = neighbourhood_coefs_neigh.idxmax()


print("R-squared on Test Set:", r2_test)
print("RMSE on Test Set:", rmse_test)
print("Coefficient on Review Scores Rating:", coef_review_score)
print("Most Expensive Property Type:", most_expensive_borough)

# R^2  = 0.046, RMSE = 140.92. The coefficent is 1.21. The most expensive bourough is Manhattan.

## 5.
train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train2 = pd.get_dummies(train[["Review Scores Rating", "Neighbourhood", "Property Type"]], drop_first=True)
X_test2 = pd.get_dummies(test[["Review Scores Rating", "Neighbourhood", "Property Type"]], drop_first=True)

X_test2 = X_test2.reindex(columns=X_train2.columns, fill_value=0)

y_train = train["Price"]
y_test = test["Price"]

model2 = LinearRegression()
model2.fit(X_train2, y_train)

y_pred2 = model2.predict(X_test2)

r2_test2 = r2_score(y_test, y_pred2)
rmse_test2 = np.sqrt(mean_squared_error(y_test, y_pred2))

coef_review_score2 = pd.Series(model2.coef_, index=X_train2.columns)["Review Scores Rating"]

property_coefs = pd.Series(model2.coef_, index=X_train2.columns)
property_coefs_only = property_coefs.filter(like="Property Type_")
most_expensive_property = property_coefs_only.idxmax()

print("R-squared on Test Set:", r2_test2)
print("RMSE on Test Set:", rmse_test2)
print("Coefficient on Review Scores Rating:", coef_review_score2)
print("Most Expensive Property Type:", most_expensive_property)

# R^2 = 0.054, and RMSE = 140.30. The coefficient = 1.2, and most expensive property to rent is Bungalow.

## 6.
# The change in coefficient from part 4 to part 5 is because there are multiple regression isolates the effect of each variable. By adding
# property type controlled the bias in the original model.

# 2.

## 1.
cars_df = pd.read_csv('cars_hw.csv')

cars_df.drop(columns=['Unnamed: 0'], inplace=True)

cars_df['Log_Price'] = np.log(cars_df['Price'])

cars_df['Arcsinh_Mileage_Run'] = np.arcsinh(cars_df['Mileage_Run'])

print(cars_df[['Price', 'Log_Price', 'Mileage_Run', 'Arcsinh_Mileage_Run']].describe())
print(cars_df[['Log_Price', 'Arcsinh_Mileage_Run']].skew())

## 2.
cars_df['Price'] = pd.to_numeric(cars_df['Price'], errors='coerce')
cars_df = cars_df.dropna(subset=['Price'])

print(cars_df['Price'].describe())

plt.figure(figsize=(8,6))
sns.kdeplot(data=cars_df, x='Price', fill=True)
plt.title('Kernel Density Plot of Price')
plt.xlabel('Price')
plt.ylabel('Density')
plt.xlim(0, 3000000)
plt.show()

price_by_make = cars_df.groupby('Make')['Price'].describe()
print(price_by_make)

make_counts = cars_df['Make'].value_counts()
popular_makes = make_counts[make_counts >= 5].index
filtered_cars_df = cars_df[cars_df['Make'].isin(popular_makes)]

top_expensive_brands = filtered_cars_df.groupby('Make')['Price'].mean().sort_values(ascending=False).head(5)

plt.figure(figsize=(12,8))
sns.kdeplot(data=filtered_cars_df, x='Price', hue='Make', common_norm=False)
plt.title('Grouped Kernel Density Plot of Price by Car Make')
plt.xlabel('Price')
plt.ylabel('Density')
plt.xlim(0, 3000000)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(x=top_expensive_brands.values, y=top_expensive_brands.index)
plt.title('Top 5 Most Expensive Car Brands (by Average Price)')
plt.xlabel('Average Price')
plt.ylabel('Car Brand')
plt.xlim(0, 2500000)
plt.show()

# The most expensive car brand is MG Motors, and the average price is between 1.5 and 2.

## 3.
train_df, test_df = train_test_split(
    filtered_cars_df,
    test_size=0.2,   # 20% test, 80% train
    random_state=42  # for reproducibility
)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")

## 4.
numeric_features = ['Make_Year', 'Mileage_Run', 'Seating_Capacity']

X_train_num = train_df[numeric_features]
y_train = train_df['Price']
X_test_num = test_df[numeric_features]
y_test = test_df['Price']

model_num = LinearRegression()
model_num.fit(X_train_num, y_train)

y_train_pred_num = model_num.predict(X_train_num)
y_test_pred_num = model_num.predict(X_test_num)

r2_train_num = r2_score(y_train, y_train_pred_num)
rmse_train_num = np.sqrt(mean_squared_error(y_train, y_train_pred_num))
r2_test_num = r2_score(y_test, y_test_pred_num)
rmse_test_num = np.sqrt(mean_squared_error(y_test, y_test_pred_num))

print("Numeric Model:")
print(f"Train R2: {r2_train_num:.4f}, Train RMSE: {rmse_train_num:.2f}")
print(f"Test R2: {r2_test_num:.4f}, Test RMSE: {rmse_test_num:.2f}")


categorical_features = ['Make', 'Color', 'Body_Type', 'No_of_Owners', 'Fuel_Type', 'Transmission', 'Transmission_Type']

X_train_cat = pd.get_dummies(train_df[categorical_features], drop_first=True)
X_test_cat = pd.get_dummies(test_df[categorical_features], drop_first=True)

X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

model_cat = LinearRegression()
model_cat.fit(X_train_cat, y_train)

y_test_pred_cat = model_cat.predict(X_test_cat)

r2_test_cat = r2_score(y_test, y_test_pred_cat)
rmse_test_cat = np.sqrt(mean_squared_error(y_test, y_test_pred_cat))

print("\nCategorical Model (one-hot):")
print(f"Test R2: {r2_test_cat:.4f}, Test RMSE: {rmse_test_cat:.2f}")


X_train_joint = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test_joint = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

model_joint = LinearRegression()
model_joint.fit(X_train_joint, y_train)

y_test_pred_joint = model_joint.predict(X_test_joint)

r2_test_joint = r2_score(y_test, y_test_pred_joint)
rmse_test_joint = np.sqrt(mean_squared_error(y_test, y_test_pred_joint))

print("\nJoint Model (numeric + categorical):")
print(f"Test R2: {r2_test_joint:.4f}, Test RMSE: {rmse_test_joint:.2f}")
print("\nPerformance Comparison:")
print(f"Improvement in R2 (Joint vs Numeric): {r2_test_joint - r2_test_num:.4f}")
print(f"Improvement in R2 (Joint vs Categorical): {r2_test_joint - r2_test_cat:.4f}")
print(f"RMSE reduction (Joint vs Numeric): {rmse_test_num - rmse_test_joint:.2f}")
print(f"RMSE reduction (Joint vs Categorical): {rmse_test_cat - rmse_test_joint:.2f}")

## 5.
results = []

for degree in range(1, 6):  # Degree 1 to 5
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train_num)
    X_test_poly = poly.transform(X_test_num)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    y_test_pred_poly = model_poly.predict(X_test_poly)

    r2_test_poly = r2_score(y_test, y_test_pred_poly)
    rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_pred_poly))

    results.append({
        'Degree': degree,
        'Test R2': r2_test_poly,
        'Test RMSE': rmse_test_poly
    })

    print(f"Degree {degree}: Test R2 = {r2_test_poly:.4f}, Test RMSE = {rmse_test_poly:.2f}")

    if r2_test_poly < 0:
        print(f"⚠️ R2 went negative at degree {degree}. Stopping search.")
        break

results_df = pd.DataFrame(results)

best_model_idx = results_df['Test R2'].idxmax()
best_degree = results_df.loc[best_model_idx, 'Degree']
best_r2 = results_df.loc[best_model_idx, 'Test R2']
best_rmse = results_df.loc[best_model_idx, 'Test RMSE']

print("\nBest Polynomial Model:")
print(f"Best Degree: {best_degree}")
print(f"Best Test R2: {best_r2:.4f}")
print(f"Best Test RMSE: {best_rmse:.2f}")
print("\nComparison to Joint Model:")
print(f"Joint Model Test R2: {r2_test_joint:.4f}")
print(f"Joint Model Test RMSE: {rmse_test_joint:.2f}")

## 6.
y_test_pred_best = model_joint.predict(X_test_joint)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.title('Predicted vs True Prices')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.show()

residuals = y_test - y_test_pred_best

plt.figure(figsize=(8,6))
sns.kdeplot(residuals, fill=True)
plt.title('Kernel Density Plot of Residuals')
plt.xlabel('Residual (True - Predicted)')
plt.ylabel('Density')
plt.axvline(0, color='red', linestyle='--')
plt.show()

print(f"Mean Residual: {np.mean(residuals):.2f}")
print(f"Standard Deviation of Residuals: {np.std(residuals):.2f}")

# The scatter plot suggesting model fits reasonably well. The residuals are approximately bell shaped, indicating no major bias.
# There are some slight skewness in the residuals suggesting the model have difficulty of predicting high priced cars.
