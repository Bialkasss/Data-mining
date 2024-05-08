import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
# Load the Titanic dataset
titanic_data = pd.read_csv("train.csv")

#
# SUMMARY STATISTICS
#

print("Summary Statistics for Numerical Features:")
print(titanic_data.describe().round(2))

#
#MISSING VALUES, UNIQUENESS
#
missing_values = titanic_data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_data.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Values Heatmap')
plt.show()

# Check the number of unique values in 'Name', 'Ticket', and 'Cabin' columns
unique_names = titanic_data['Name'].nunique()
unique_tickets = titanic_data['Ticket'].nunique()
unique_cabins = titanic_data['Cabin'].nunique()

print("Number of unique values in 'Name':", unique_names)
print("Number of unique values in 'Ticket':", unique_tickets)
print("Number of unique values in 'Cabin':", unique_cabins)


#
#CORREALTION
#


# heatmap
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
plt.figure(figsize=(5,4))
sns.heatmap(abs(titanic_data[numerical_features].corr()), cmap='magma')


#
#HISTOGRAM OF ALL FEATURES THAT CAN MATTER
#


# Select only input features (excluding PassengerId and Survived)
input_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
input_data = titanic_data[input_features]

# Plotting
plt.figure(figsize=(15, 12))
for i, feature in enumerate(input_features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(titanic_data[feature], kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

palet = ["#b0438d", "#f0eda3", "#8b5ed6", "#290b5c"]
import seaborn as sns
import matplotlib.pyplot as plt

# Define input features
input_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Separate numerical and categorical features
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Plotting numerical features
plt.figure(figsize=(15, 12))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(titanic_data[feature], kde=True, color="#b0438d", palette=palet)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Plotting categorical features
plt.figure(figsize=(15, 12))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=feature, data=titanic_data, palette=palet)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()



#
#BOXPLOTS
#


# Pairplot for numerical features #USELESS
sns.pairplot(titanic_data, vars=numerical_features, hue='Survived', kind='scatter', palette='Set1')
plt.suptitle('Pairplot of Numerical Features in Relation to Survival', y=1.02)
plt.show()

# Box plots for numerical features in relation to 'Survived'
plt.figure(figsize=(30, 12))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='Survived', y=feature, data=titanic_data, palette='Set1')
    plt.title(f'{feature} vs. Survived')
    plt.xlabel('Survived')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
palet = ["#b0438d", "#f0eda3"]

# Bar plots for categorical features in relation to 'Survived'
categorical_features = ['Sex', 'Embarked']
plt.figure(figsize=(12, 6))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 2, i)
    sns.barplot(x=feature, y='Survived', data=titanic_data, palette=palet)
    plt.title(f'Survival Rate by {feature}')
    plt.ylabel('Survival Rate')
    plt.xlabel(feature)
plt.figure(figsize=(12, 6))
sns.barplot(x='Embarked', y='Survived', data=titanic_data, palette=palet)
plt.title(f'Survival Rate by Embarked')
plt.ylabel('Survival Rate')
plt.xlabel('Embarked')
plt.show()

# Create boxplot for Fare with limited y-scale
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=titanic_data, palette=palet)
plt.title('Fare vs. Survived')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.ylim(0, 300)  # Limit y-axis scale to 300

plt.tight_layout()
plt.show()


############################################################
# Extract counts of 'Survived' categories
survived_counts = titanic_data['Survived'].value_counts()

# Define labels and colors
labels = ['Survived', 'Not Survived']
sizes = [survived_counts[1], survived_counts[0]]
colors = ["#b0438d", "#f0eda3"]  # Custom color palette

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Survival Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.show()

# Scatter plot of Pclass vs Fare
plt.figure(figsize=(8, 6))
sns.scatterplot(data=titanic_data, x='Pclass', y='Age', s=100, alpha=0.5)
plt.title('Scatter Plot of Pclass vs Fare')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.show()

print(max(titanic_data['Fare']))

# Define numerical features
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 'Survived' vs each numerical feature
for idx, feature in enumerate(numerical_features):
    row = idx // 2
    col = idx % 2
    sns.boxplot(x='Survived', y=feature, data=titanic_data, ax=axes[row, col], palette=palet)
    axes[row, col].set_title(f'Survived vs {feature}')
    axes[row, col].set_xlabel('Survived')
    axes[row, col].set_ylabel(feature)

# Adjust layout
plt.tight_layout()
plt.show()