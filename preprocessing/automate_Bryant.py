import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # pilih kolom yang dipakai
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

    # handle missing value
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # encoding
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    preprocess_data(
        "titanic_data/train.csv",
        "titanic_preprocessing/titanic_clean.csv"
    )
