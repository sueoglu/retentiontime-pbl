import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    data = pd.read_csv('/Users/oykusuoglu/PythonProjects/RetentionTimePBL/normalized_pw_dataset.csv')
except FileNotFoundError:
    print("Error: file not found")
    data = None
except pd.errors.EmptyDataError:
    print("Error: empty file.")
    data = None
except pd.errors.ParserError:
    print("Error: csv file parsing")
    data = None


if data is not None:
    print("preview:")
    print(data.head())

    if 'sequence' in data.columns:
        data = data.drop(columns=['sequence'])

    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("error")
    else:
        try:
            corr_matrix = numeric_data.corr(method="pearson")

            plt.figure(figsize=(15, 15))
            heatmap = sns.heatmap(corr_matrix,
                                  annot=True,
                                  cmap='viridis',
                                  vmin=-1,
                                  vmax=1,
                                  center=0,
                                  square=True,
                                  fmt='.3f',
                                  annot_kws={'size': 10},
                                  cbar_kws={'label': 'Correlation Coefficient'}
                                  )
            plt.title('Correlation Analysis of Molecular Properties\nand Retention Time')
            plt.tight_layout()
            plt.show()
            print(corr_matrix)
        except Exception as e:
            print(f"Error:{e}")

