import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#////////// Preproccesing DATA  ////////////////
correctdata = False
df = pd.read_csv('ds.csv')
df = pd.DataFrame(df)
total_rows_dropped = 0

while correctdata == False:
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
    plt.title("Relationship Matrix")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(df_matriz)
    plt.title("Scatter Plot")
    plt.show()

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  
    pd.set_option('display.max_colwidth', None)

    negative_values = df[df['Y'] < 0]
    negative_values_indices = df[df['Y'] < 0].index
    alteration_indices_may = df[(df['X1'] >= 0.75) & (df['X2'] >= 0.7) & (df['Y'] <= 0.6)].index
    alteration_indices_min = df[(df['X1'] < 0.75) & (df['X2'] < 0.7) & (df['Y'] >= 0.5)].index

    #if not negative_values.empty:
        #print("Values below 0 found in the following rows:")
        #for index, row in negative_values.iterrows():
            #print(f"Row {index}: X1 = {row['X1']}, X2 = {row['X2']}")
    
    if not alteration_indices_may.empty:
        print("Data alteration detected in the following rows:")
        for index in alteration_indices_may:
            row = df.loc[index]
            #print(f"Row {index}: X1 = {row['X1']}, X2 = {row['X2']}, Y = {row['Y']}")

    elif not alteration_indices_min.empty:
        print("Data alteration detected in the following rows:")
        for index in alteration_indices_min:
            row = df.loc[index]
            #print(f"Row {index}: X1 = {row['X1']}, X2 = {row['X2']}, Y = {row['Y']}")

    else:
        print("Correct data ... ")
        print(f"Total rows dropped during preprocessing: {total_rows_dropped}")
        correctdata = True
    
    rows_to_drop = len(negative_values_indices) + len(alteration_indices_may) + len(alteration_indices_min)
    total_rows_dropped += rows_to_drop
    df.drop(negative_values_indices, inplace=True)
    df.drop(alteration_indices_may, inplace=True)
    df.drop(alteration_indices_min, inplace=True)

#////////// Preproccesing DATA  ////////////////

