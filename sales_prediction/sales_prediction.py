import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\sales_prediction\Advertising.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Visualize relationships
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg', height=4)
plt.suptitle('Ad Spend vs Sales', y=1.02)
plt.tight_layout()
plt.show()

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation (no 'squared' keyword issue)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print performance
print("\nModel Performance:")
print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Show coefficients
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Impact on Sales:")
print(coeff_df)

# Plot feature impact
sns.barplot(data=coeff_df, x='Coefficient', y='Feature', palette='mako')
plt.title("Advertising Spend Impact on Sales")
plt.tight_layout()
plt.show()

# Optional: Predict future sales based on planned ad budgets
future_ads = pd.DataFrame({
    'TV': [100, 150],
    'Radio': [20, 25],
    'Newspaper': [10, 15]
})
future_sales = model.predict(future_ads)
print("\nPredicted Future Sales for planned campaigns:")
print(future_sales)

# Insights
print("\n📊 Insights:")
print("- TV and Radio have the strongest impact on sales.")
print("- Newspaper ads contribute very little to prediction accuracy.")
print("- The model can help plan ad budgets effectively for future sales targets.")
