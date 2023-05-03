
## Need to clean code / Put into a Jnotebook

#%%
import seaborn as sns
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
alt.data_transformers.disable_max_rows()

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
df.head()
#%%
yes = df.loc[df["y"]== "yes"]
#%%
no = df.loc[df["y"]== "no"]

#%%
# This is for jobs who say yes:
response_counts = yes.groupby("job")['y'].value_counts().unstack()
response_counts = response_counts.reset_index()
#%%
#jobs that say no:
negative_response = no.groupby('job')['y'].value_counts().unstack()
negative_response = negative_response.reset_index()
#%%
joby_chart = alt.Chart(response_counts).mark_bar().encode(
    x= "job",
    y = "yes"
)

jobn_chart = alt.Chart(negative_response).mark_bar().encode(
    x= "job",
    y = "no"
)



# %%
response_counts = df.groupby("job")['y'].value_counts().unstack().fillna(0)
response_counts['total'] = response_counts['yes'] + response_counts['no']
response_counts['yes_percentage'] = response_counts['yes'] / response_counts['total'] * 100

# Reset the index
response_counts = response_counts.reset_index()

# Create an Altair chart using the calculated percentage
percentagejobs_chart = alt.Chart(response_counts).mark_bar().encode(
    x='job',
    y='yes_percentage'
)

#%%
source = df

matrix = alt.Chart(source).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='y:N'
).properties(
    width=150,
    height=150
).repeat(
    row=['age','job','marital','education'],
    column=['education', 'marital', 'job','age']
).interactive()

matrix.show()


#%%

jobn_chart.show()
joby_chart.show()
percentagejobs_chart.show()
# %%
source = df

matrix = alt.Chart(source).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='y:N'  # Use the target variable 'y' for coloring the circles
).properties(
    width=150,
    height=150
).repeat(
    row=['age', 'campaign','cons.conf.idx','euribor3m','nr.employed','cons.price.idx'],
    column=['age', 'campaign','cons.conf.idx','euribor3m','nr.employed','cons.price.idx']
).interactive()

matrix.show()
matrix.show()


# %%
source = df

matrix = alt.Chart(source).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='y:N'  # Use the target variable 'y' for coloring the circles
).properties(
    width=150,
    height=150
).facet(
    row=alt.Row("row:N", sort=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']),
    column=alt.Column("column:N", sort=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']),
    data=df.melt(id_vars=['y'], value_vars=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
                 var_name='column', value_name='value'),
    title="Scatterplot Matrix of Quantitative Variables"
).interactive()

matrix.show()
# %%
heat = sns.heatmap(df.corr())
plt.show(block=True)
# %%
