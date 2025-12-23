import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#import pandas as pd
#import numpy as np
from sklearn.preprocessing import LabelEncoder
#import warnings
#warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os



# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("PART 1: DATA LOADING & EXPLORATION")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📂 Loading data...")
df = pd.read_csv('jay/data/Telco-Churn.csv')

print(f"✅ Data loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: INITIAL EXPLORATION
# ============================================================
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary (Numerical columns):")
print(df.describe())

print("\n❓ Checking for Missing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("   Missing values found:")
    print(missing[missing > 0])
else:
    print("   ✅ No missing values found!")

# ============================================================
# STEP 3: ANALYZE TARGET VARIABLE (Churn)
# ============================================================
print("\n🎯 Target Variable Distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)

churn_rate = (churn_counts['Yes'] / len(df) * 100)
print(f"\n📊 Churn Rate: {churn_rate:.2f}%")
print(f"   - {churn_counts['Yes']} customers left (churned)")
print(f"   - {churn_counts['No']} customers stayed")

# ============================================================
# STEP 4: EXPLORE CATEGORICAL COLUMNS
# ============================================================
print("\n📋 Categorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"   Found {len(categorical_cols)} categorical columns:")
for col in categorical_cols:
    print(f"      - {col}: {df[col].nunique()} unique values")

# ============================================================
# STEP 5: EXPLORE NUMERICAL COLUMNS
# ============================================================
print("\n🔢 Numerical Columns:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"   Found {len(numerical_cols)} numerical columns:")
for col in numerical_cols:
    print(f"      - {col}")

print("\n" + "="*60)
print("✨ PART 1 COMPLETE - Data Exploration Done!")
print("="*60)
print("\n💡 Key Findings:")
print(f"   - Total customers: {len(df)}")
print(f"   - Churn rate: {churn_rate:.2f}%")
print(f"   - Features: {len(df.columns) - 1}")
print("\n➡️  Ready for Part 2: Data Cleaning & Feature Engineering")









print("="*60)
print("PART 2: DATA CLEANING & FEATURE ENGINEERING")
print("="*60)

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
print("\n📂 Loading data...")
df = pd.read_csv('jay/data/Telco-Churn.csv')
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
print("\n🧹 Starting Data Cleaning...")

# Fix TotalCharges column
print("\n   Fixing TotalCharges column...")
print(f"   Current data type: {df['TotalCharges'].dtype}")

# Convert to numeric (some values are spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many missing values we created
nan_count = df['TotalCharges'].isna().sum()
if nan_count > 0:
    print(f"   ⚠️  Found {nan_count} invalid values")
    print(f"   📝 Filling with median value...")
    median_value = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_value, inplace=True)
    print(f"   ✅ Filled with median: ${median_value:.2f}")
else:
    print(f"   ✅ No issues found!")

# Drop customerID (not useful for prediction)
print("\n   Removing customerID column...")
df = df.drop('customerID', axis=1)
print(f"   ✅ Dropped! New shape: {df.shape}")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n⚙️  Creating New Features...")

# 1. Tenure Groups
print("\n   1️⃣  Creating tenure_group...")
df['tenure_group'] = pd.cut(df['tenure'], 
                             bins=[0, 12, 24, 48, 72], 
                             labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'])
print("      ✅ Grouped customers by how long they've stayed")
print(f"      Distribution:\n{df['tenure_group'].value_counts().sort_index()}")

# 2. Monthly Charges Groups
print("\n   2️⃣  Creating monthly_charges_group...")
df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'], 
                                      bins=[0, 35, 65, 100, 150],
                                      labels=['Low', 'Medium', 'High', 'Very High'])
print("      ✅ Grouped customers by monthly payment amount")
print(f"      Distribution:\n{df['monthly_charges_group'].value_counts().sort_index()}")

# 3. Average Monthly Charges
print("\n   3️⃣  Creating avg_monthly_charges...")
df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
print("      ✅ Calculated: TotalCharges / tenure")
print(f"      Average: ${df['avg_monthly_charges'].mean():.2f}")

# ============================================================
# STEP 4: ENCODING CATEGORICAL VARIABLES
# ============================================================
print("\n🔤 Encoding Categorical Variables...")

# Separate features and target
print("\n   Separating features (X) and target (y)...")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Encode target variable
print("\n   Encoding target variable (Churn)...")
y = y.map({'Yes': 1, 'No': 0})
print("      ✅ Yes → 1 (Churned)")
print("      ✅ No → 0 (Stayed)")
print(f"      Churn distribution: {y.value_counts().to_dict()}")

# Get categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n   Found {len(categorical_cols)} categorical columns to encode")

# Binary encoding (for columns with only 2 values)
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

print("\n   📝 Label Encoding (for binary columns)...")
le = LabelEncoder()
for col in binary_cols:
    if col in X.columns:
        original_values = X[col].unique()[:2]  # Show first 2 values
        X[col] = le.fit_transform(X[col])
        print(f"      ✅ {col}: {original_values[0]}→0, {original_values[1]}→1")

# One-Hot Encoding (for columns with 3+ values)
print("\n   🔢 One-Hot Encoding (for multi-category columns)...")
multi_category_cols = [col for col in categorical_cols if col not in binary_cols]

print(f"      Encoding {len(multi_category_cols)} columns:")
for col in multi_category_cols:
    print(f"         - {col} ({X[col].nunique()} categories)")

X = pd.get_dummies(X, columns=multi_category_cols, drop_first=True, dtype=int)
print(f"\n      ✅ Done! New shape: {X.shape}")
print(f"      Total features increased from {len(df.columns)-1} to {X.shape[1]}")

# ============================================================
# STEP 5: SAVE CLEANED DATA
# ============================================================
print("\n💾 Saving Cleaned Data...")

# Save to new files
X.to_csv('jay/data/X_cleaned.csv', index=False)
y.to_csv('jay/data/y_cleaned.csv', index=False)

print("   ✅ Saved:")
print("      - jay/data/X_cleaned.csv")
print("      - jay/data/y_cleaned.csv")

# ============================================================
# STEP 6: SUMMARY
# ============================================================
print("\n" + "="*60)
print("✨ PART 2 COMPLETE - Data Cleaning Done!")
print("="*60)
print(f"📊 Summary:")
print(f"   - Original features: {len(df.columns) - 1}")
print(f"   - New features created: 3 (tenure_group, monthly_charges_group, avg_monthly_charges)")
print(f"   - After encoding: {X.shape[1]} features")
print(f"   - Target encoded: Yes=1, No=0")
print(f"   - Ready for: Scaling & Train-Test Split")
print("="*60)

# Show first few rows of cleaned data
print("\n👀 Preview of cleaned features (first 5 rows, first 10 columns):")
print(X.iloc[:5, :10])

print("\n👀 Preview of target variable (first 10 values):")
print(y.head(10).tolist())

print("\n➡️  Ready for Part 3: Scaling & Train-Test Split!")








print("="*60)
print("PART 3: SCALING & TRAIN-TEST SPLIT")
print("="*60)

# ============================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================
print("\n📂 Loading cleaned data...")
X = pd.read_csv('jay/data/X_cleaned.csv')
y = pd.read_csv('jay/data/y_cleaned.csv')

print(f"✅ Loaded!")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# ============================================================
# STEP 2: FEATURE SCALING
# ============================================================
print("\n📏 Scaling Numerical Features...")

# Identify numerical columns that need scaling
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges']

print(f"\n   Columns to scale: {len(numerical_cols)}")
for col in numerical_cols:
    print(f"      - {col}")

# Show before scaling
print(f"\n   📊 Before Scaling (sample values):")
print(X[numerical_cols].head(3))

# Create and fit scaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Show after scaling
print(f"\n   📊 After Scaling (sample values):")
print(X[numerical_cols].head(3))

print("\n   ✅ Scaling complete!")
print("      Why? So big numbers don't dominate small numbers in the model")

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================
print("\n✂️  Splitting Data into Training and Testing Sets...")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Keep same churn ratio in both sets
)

print(f"\n   ✅ Split complete!")
print(f"\n   📊 Training Set:")
print(f"      - Samples: {X_train.shape[0]}")
print(f"      - Features: {X_train.shape[1]}")
print(f"      - Churn rate: {(y_train['Churn'].sum() / len(y_train) * 100):.2f}%")

print(f"\n   📊 Test Set:")
print(f"      - Samples: {X_test.shape[0]}")
print(f"      - Features: {X_test.shape[1]}")
print(f"      - Churn rate: {(y_test['Churn'].sum() / len(y_test) * 100):.2f}%")

print("\n   💡 Why split?")
print("      - Train set: Teach the model")
print("      - Test set: Evaluate how well it learned (unseen data)")

# ============================================================
# STEP 4: SAVE EVERYTHING
# ============================================================
print("\n💾 Saving Processed Data...")

# Create models directory if it doesn't exist
os.makedirs('jay/models', exist_ok=True)

# Save train-test split
X_train.to_csv('jay/data/X_train.csv', index=False)
X_test.to_csv('jay/data/X_test.csv', index=False)
y_train.to_csv('jay/data/y_train.csv', index=False)
y_test.to_csv('jay/data/y_test.csv', index=False)

# Save the scaler (we'll need it later for predictions)
joblib.dump(scaler, 'jay/models/scaler.pkl')

print("\n   ✅ Saved successfully:")
print("      📁 Data files:")
print("         - jay/data/X_train.csv")
print("         - jay/data/X_test.csv")
print("         - jay/data/y_train.csv")
print("         - jay/data/y_test.csv")
print("      📁 Model files:")
print("         - jay/models/scaler.pkl")

# ============================================================
# STEP 5: FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✨ PREPROCESSING COMPLETE - READY FOR MODEL TRAINING!")
print("="*60)

print("\n📊 Final Dataset Summary:")
print(f"   Original data: 7,043 customers")
print(f"   Features: {X_train.shape[1]}")
print(f"   Training samples: {X_train.shape[0]} (80%)")
print(f"   Testing samples: {X_test.shape[0]} (20%)")
print(f"   Churn rate: ~26.5% (imbalanced)")

print("\n🎯 What We Did:")
print("   ✅ Cleaned data (fixed TotalCharges)")
print("   ✅ Created new features (tenure_group, etc.)")
print("   ✅ Encoded text to numbers")
print("   ✅ Scaled numerical features")
print("   ✅ Split into train/test sets")
print("   ✅ Saved everything for model training")

print("\n🚀 Next Steps:")
print("   Phase 2: Build and train ML models")
print("   We'll try: Logistic Regression, Random Forest, XGBoost")
print("   Goal: Predict which customers will churn!")

print("\n" + "="*60)
print("🎉 PHASE 1 COMPLETE!")
print("="*60)

# Show feature names for reference
print("\n📋 All 37 Features (for reference):")
for i, col in enumerate(X.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n✅ Everything is ready! You can now move to Phase 2.")