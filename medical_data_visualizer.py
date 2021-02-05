import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = list(map(lambda w, h: 1 if w/((h/100)**2) > 25 else 0, df['weight'], df['height']))

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

def normalizer(value):
	newvalue = value - 1
	if newvalue > 1:
		newvalue = 1
	return newvalue

df['cholesterol'] = list(map(normalizer, df['cholesterol']))
df['gluc'] = list(map(normalizer, df['gluc']))

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
	legal_columns = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
	df_cat = df
	for item in df.items():
		col = item[0]
		if col not in legal_columns:
			df_cat = df_cat.drop(col, 1)
	df_cat = pd.melt(df_cat)


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
	df_cat['cardio'] = df['cardio'].tolist()*6
	df_cat = df_cat.sort_values(by=['cardio'])
	print(df_cat)
	totalrows = df_cat.shape[0]
	cardioindex = df_cat['cardio'].to_list().index(1)
	df_cat = df_cat.head(cardioindex).groupby(['variable']).sum().append(df_cat.tail(totalrows-cardioindex).groupby(['variable']).sum())
	df_cat['cardio'] = list(map(lambda x: int(x if x == 0 else x/abs(x)), df_cat['cardio']))
	df_cat['name'] = df_cat.index.values
	df_cat['amount'] = df_cat['value']
	df_cat['value'] = [1 for thing in df_cat['name']]
	df_cat2 = df_cat.copy()
	df_cat2['value'] = [0 for thing in df_cat2['name']]
	df_cat2['amount'] = [int(cardioindex/6)-df_cat['amount'][i] if df_cat['cardio'][i] == 0 else int((totalrows-cardioindex)/6)-df_cat['amount'][i] for i in range(len(df_cat['amount']))]
	df_cat = df_cat.append(df_cat2)
	df_cat['variable'] = df_cat['name']
	df_cat['total'] = df_cat['amount']
	print(df_cat)


    # Draw the catplot with 'sns.catplot()'

	fig = sns.catplot(data = df_cat, x = 'variable', y = 'total', col = 'cardio', kind = 'bar', hue = 'value')
	fig = fig.fig


    # Do not modify the next two lines
	fig.savefig('catplot.png')
	return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
	df_heat = df.copy()
	df_heat['ap_lo'] = list(map(lambda x, y: None if x > y else x, df_heat['ap_lo'], df_heat['ap_hi']))
	toProcess = ['height', 'weight']
	for thing in toProcess:
		low, high = df_heat[thing].quantile(0.025), df_heat[thing].quantile(0.975)
		df_heat[thing] = list(map(lambda x: None if x < low or x > high else float(x), df_heat[thing]))
	df_heat.dropna(inplace = True)


    # Calculate the correlation matrix
	corr = df_heat.corr().round(1)

    # Generate a mask for the upper triangle
	mask = np.array([[(x>=y) for x in range(corr.shape[0])] for y in range(corr.shape[0])])

    # Set up the matplotlib figure
	fig, ax = plt.subplots(figsize=(7,7))

    # Draw the heatmap with 'sns.heatmap()'

	sns.heatmap(corr, annot = True, mask = mask, fmt = '.1f')


    # Do not modify the next two lines
	fig.savefig('heatmap.png')
	return fig
