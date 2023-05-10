#%%
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import altair as alt

#%%
url_bank = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

bank = pd.read_csv(url_bank)
#bank.replace({'unknown': np.nan, 999: np.nan, 'nonexistent': np.nan}, inplace=True)


#%%
bank.columns

#%%
bank['day_of_week'].value_counts()

#%%
# Filter bank DataFrame to keep only rows where 'contact' is 'cellular'
filtered_bank = bank.loc[(bank['contact'] == 'cellular') & (bank['marital'] != 'unknown') & (bank['job'] == 'student')]

# Compute value counts on the filtered DataFrame
counts = filtered_bank[['day_of_week', 'marital', 'y']].value_counts().reset_index()
counts.columns = ['day_of_week', 'marital', 'y', 'count']

# Define the desired order of the categories
day_order = ['mon', 'tue', 'wed', 'thu', 'fri']

# Apply the order to the day_of_week column of the counts DataFrame
counts = counts.sort_values(by=['day_of_week'], key=lambda x: pd.Categorical(x, categories=day_order))

# Create an Altair chart with the sorted x-axis
chart1 = alt.Chart(counts).mark_bar().encode(
    x=alt.X('day_of_week:N', sort=day_order),
    y='count:Q',
    color='y:N'#,
    #column='marital:N'
).properties(
    width=100,
    title="Data Distribution by Day of Week, Marital Status, and Y (Cellular Contacts Only)"
).facet(facet='marital:N')

# Display the chart
chart1

#%%
# Filter bank DataFrame to keep only rows where 'contact' is 'cellular'
filtered_bank2 = bank.loc[(bank['contact'] == 'telephone') & (bank['marital'] != 'unknown')]

# Compute value counts on the filtered DataFrame
counts = filtered_bank2[['day_of_week', 'marital', 'y']].value_counts().reset_index()
counts.columns = ['day_of_week', 'marital', 'y', 'count']

# Define the desired order of the categories
day_order = ['mon', 'tue', 'wed', 'thu', 'fri']

# Apply the order to the day_of_week column of the counts DataFrame
counts = counts.sort_values(by=['day_of_week'], key=lambda x: pd.Categorical(x, categories=day_order))

# Create an Altair chart with the sorted x-axis
chart2 = alt.Chart(counts).mark_bar().encode(
    x=alt.X('day_of_week:N', sort=day_order),
    y='count:Q',
    color='y:N',
    column='marital:N'
).properties(
    width=100,
    title="Data Distribution by Day of Week, Marital Status, and Y (Telephone Contacts Only)"
)#.facet(facet='marital:N')

# Display the chart
chart2


#%%
# Filter the DataFrame by 'cellular' or 'telephone' contact
filtered_bank2 = bank.loc[(bank['contact'] == 'cellular') | (bank['contact'] == 'telephone')]



# Compute the success rate of calling customers by contact type and job
success_rate = filtered_bank2.groupby(['contact', 'job'])['y'].value_counts(normalize=True).reset_index(name='success_rate')
success_rate = success_rate.loc[success_rate['y'] == 'yes']

# Create an Altair chart with the success rate
chart3 = alt.Chart(success_rate).mark_bar().encode(
    x=alt.X('job:N', axis=alt.Axis(title='Job Type')),
    y=alt.Y('success_rate:Q', axis=alt.Axis(format='.0%')),
    color=alt.Color('contact:N', legend=None),
    column=alt.Column('contact:N', header=alt.Header(title='Contact Type')),
    tooltip=[alt.Tooltip('success_rate:Q', format='.2%')],
).properties(
    width=200,
    title="Success Rate of Calling Customers by Contact Type and Job"
)

# Display the chart
chart3


#%%

# Filter the DataFrame by 'cellular' or 'telephone' contact
filtered_bank2 = bank.loc[((bank['contact'] == 'cellular') | (bank['contact'] == 'telephone'))
                          & (bank['job'] != 'unknown')]

# Count the number of responses for each combination of contact type and job
response_counts = filtered_bank2.groupby(['contact', 'job', 'y'])['y'].count().reset_index(name='count')

# Create an Altair chart with the response counts
chart4 = alt.Chart(response_counts).mark_bar().encode(
    x=alt.X('job:N', axis=alt.Axis(title='Job Type')),
    y=alt.Y('count:Q', axis=alt.Axis(title='Response Count')),
    color=alt.Color('y:N', legend=alt.Legend(title='Response')),
    column=alt.Column('contact:N', header=alt.Header(title='Contact Type')),
    tooltip=[alt.Tooltip('count:Q')],
).properties(
    width=200,
    title="Responses by Contact Type and Job"
)

# Display the chart
chart4

#%%

# Filter the DataFrame by 'cellular' or 'telephone' contact
filtered_bank2 = bank.loc[(((bank['contact'] == 'cellular') | (bank['contact'] == 'telephone')) & (bank['job'] != 'unknown')) & (bank['job'] == 'retired')]
filtered_bank2[['job', 'y']].value_counts()

#%%
# Group the filtered dataset by 'job' and 'y' columns and count the number of 'yes' and 'no' responses for each job
counts = filtered_bank2.groupby(['job', 'y'])['y'].count().reset_index(name='count')

# Pivot the counts to create a new DataFrame with the 'yes' and 'no' counts for each job
counts_pivot = counts.pivot(index='job', columns='y', values='count').reset_index()

# Calculate the proportion of 'yes' responses to 'no' responses for each job
counts_pivot['proportion'] = counts_pivot['yes'] / counts_pivot['no']

# Create an Altair chart with the proportions of 'yes' responses to 'no' responses by job
chart5 = alt.Chart(counts_pivot).mark_bar().encode(
    x=alt.X('job:N', axis=alt.Axis(title='Job Type')),
    y=alt.Y('proportion:Q', axis=alt.Axis(title='Proportion of "Yes" Responses to "No" Responses')),
    color=alt.Color('job:N'),
    tooltip=[alt.Tooltip('proportion:Q', format='.2f')],
).properties(
    width=400,
    title="Proportion of 'Yes' Responses to 'No' Responses by Job Type"
)

# Display the chart
chart5
#%%

# Filter the DataFrame by 'cellular' or 'telephone' contact
# Filter the DataFrame by 'cellular' or 'telephone' contact
filtered_bank3 = bank.loc[(bank['contact'] == 'cellular') | (bank['contact'] == 'telephone')]

# Group the filtered dataset by 'previous' and 'y' columns and count the number of 'yes' and 'no' responses for each contact frequency
counts = filtered_bank3.groupby(['previous', 'y'])['y'].count().reset_index(name='count')

# Create an Altair chart with the number of times contacted on the x-axis and the count of 'yes' and 'no' responses on the y-axis
chart6 = alt.Chart(counts).mark_bar().encode(
    x=alt.X('previous:Q', axis=alt.Axis(title='Number of Times Contacted')),
    y=alt.Y('count:Q', axis=alt.Axis(title='Count')),
    color=alt.Color('y:N', scale=alt.Scale(range=['#CC0000', '#00CC00']), legend=alt.Legend(title='Response')),
    tooltip=[alt.Tooltip('count:Q', title='Count'), alt.Tooltip('y:N', title='Response')],
).properties(
    width=400,
    title="Count of 'Yes' and 'No' Responses by Number of Times Contacted"
)

# Display the chart
chart6

#%%


#%%
#Make me an altair chart that shows the proportion of 'yes' responses to 'no' responses by number of times contacted
# Filter the DataFrame by 'cellular' or 'telephone' contact
filtered_bank3 = bank.loc[(bank['contact'] == 'cellular') | (bank['contact'] == 'telephone')]
filtered_bank3[['previous', 'y']].value_counts()

#%%
# Group the filtered dataset by 'previous' and 'y' columns and count the number of 'yes' and 'no' responses for each contact frequency
counts = filtered_bank3.groupby(['previous', 'y'])['y'].count().reset_index(name='count')

# Pivot the counts to create a new DataFrame with the 'yes' and 'no' counts for each contact frequency
counts_pivot = counts.pivot(index='previous', columns='y', values='count').reset_index()

# Calculate the proportion of 'yes' responses to 'no' responses for each contact frequency
counts_pivot['proportion'] = counts_pivot['yes'] / counts_pivot['no']

# Create an Altair chart with the proportions of 'yes' responses to 'no' responses by number of times contacted
chart7 = alt.Chart(counts_pivot).mark_bar().encode(
    x=alt.X('previous:Q', axis=alt.Axis(title='Number of Times Contacted'), scale=alt.Scale(domain=(0, 7))),
    y=alt.Y('proportion:Q', axis=alt.Axis(title='Proportion of "Yes" Responses to "No" Responses')),
    color=alt.Color('previous:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title='Number of Times Contacted')),
    tooltip=[alt.Tooltip('proportion:Q', format='.2f')],
).properties(
    width=400,
    title="Proportion of 'Yes' Responses to 'No' Responses by Number of Times Contacted"
)

# Display the chart
chart7

# %%
