import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#////////////// PREPROCESSING STEP ///////////////////////

correctdata = False
df = pd.read_csv('DataANN2.csv')
total_rows_dropped = 0

while not correctdata:
    rows_dropped_this_iteration = 0 
    negative_values_indices = df[df['Y'] < 0].index
    alteration_indices_may = df[(df['X1'] >= 0.88) & (df['X2'] >= 0.2) & (df['Y'] <= 0.25)].index
    alteration_indices_min = df[(df['X1'] < 0.85) & (df['X2'] <= 0.1) & (df['Y'] >= 0.25)].index
    duplicated_rows = df[df.duplicated(subset=['X1', 'X2'])].index

    u_rows = sorted(df['X1'].unique())
    u_cols = df['X2'].unique()
    matriz = np.empty((len(u_rows), len(u_cols)))
    matriz[:] = np.nan
    for index, row in df.iterrows():
        y_index = np.where(u_rows == row['X1'])[0][0]
        x_index = np.where(u_cols == row['X2'])[0][0]
        matriz[y_index, x_index] = row['Y']
    df_matriz = pd.DataFrame(matriz, index=u_rows, columns=u_cols)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_matriz, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Relation Matrix")
    plt.show()

    if not negative_values_indices.empty:
        df = df.drop(negative_values_indices)
        rows_dropped_this_iteration += len(negative_values_indices)
    elif not alteration_indices_may.empty:
        df = df.drop(alteration_indices_may)
        rows_dropped_this_iteration += len(alteration_indices_may)
    elif not alteration_indices_min.empty:
        df = df.drop(alteration_indices_min)
        rows_dropped_this_iteration += len(alteration_indices_min)
    elif not duplicated_rows.empty:
        df = df.drop(duplicated_rows)
        rows_dropped_this_iteration += len(duplicated_rows)
    else:
        print("Correct data .... ")
        print(f"Total eliminated Data: {total_rows_dropped}")
        correctdata = True

    total_rows_dropped += rows_dropped_this_iteration
    print(f"Eliminated data each i: {rows_dropped_this_iteration}")

#df.to_csv('data_clean.csv', index=False)

#///////////////////// CLEAN DATA ////////////////////////////

#///////////////////// DATA SELECION PROCESS ////////////////////////////

df_clean = pd.read_csv('data_clean.csv')

min_X1_row = df_clean['X1'].idxmin()
max_X1_row = df_clean['X1'].idxmax()
min_X2_row = df_clean['X2'].idxmin()
max_X2_row = df_clean['X2'].idxmax()
special_rows = df.loc[[min_X1_row, max_X1_row, max_X2_row]]
df_clean = df_clean.drop([min_X1_row, max_X1_row, max_X2_row], axis=0)

train_data, test_data = train_test_split(df_clean, test_size=0.32, random_state=42)
train_data = pd.concat([train_data, special_rows])

print("Training Data:")
print(train_data)
print(train_data.shape)

print("\nTest Data:")
print(test_data)
print(test_data.shape)

#train_data.to_csv('data_train.csv', index=False)
#test_data.to_csv('data_test.csv', index=False)

#///////////////////// SELECTED DATA ////////////////////////////

