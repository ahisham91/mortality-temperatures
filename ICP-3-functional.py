import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.formula.api as smf
 
PATH = r'C:\Users\anish\Downloads'
fname = 'icp_indiv_2_dg2011_rep_nomiss.csv'
fname2 = 'icp_indiv_2_county_avetemp.csv'
df_main = pd.read_csv(os.path.join(PATH, fname))
df_main.head()

#Q1a
temp_vars = ['tday_lt10', 'tday_10_20', 'tday_20_30', 'tday_30_40', 'tday_40_50', 
             'tday_50_60', 'tday_60_70', 'tday_70_80', 'tday_80_90', 'tday_gt90']
data_temp = df_main[['population'] + temp_vars]
data_temp
weighted_avgs = {}
for var in temp_vars: 
    weighted_avgs[var] = (data_temp[var] * data_temp['population']).sum() / data_temp['population'].sum()
temp_day_vars = ['tday_lt10', 'tday_10_20', 'tday_20_30', 'tday_30_40', 'tday_40_50', 
                 'tday_50_60', 'tday_60_70', 'tday_70_80', 'tday_80_90', 'tday_gt90']
temp_day_labels = ["<10°F", "10-20°F", "20-30°F", "30-40°F", "40-50°F", "50-60°F", "60-70°F", "70-80°F", "80-90°F", ">90°F"]
temp_day_values = [weighted_avgs[var] for var in temp_day_vars]
plt.figure(figsize=(12, 6))
plt.bar(temp_day_labels, temp_day_values, color='skyblue')
plt.xlabel('Temperature Range')
plt.ylabel('Average Number of Days')
plt.title('Population-Weighted Average Number of Days (County-by-Year) in Each Temperature Range')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

#Q1b
average_days_above_90F = weighted_avgs['tday_gt90']
average_days_above_90F
annual_data = df_main.groupby('year').apply(lambda x: (x['tday_gt90'] * x['population']).sum() / x['population'].sum())
average_days_above_90F_per_year = annual_data.mean()
average_days_above_90F_per_year

 #Q1c
max_days_above_90F = df_main.groupby('county')['tday_gt90'].mean().idxmax()
max_days_value = df_main.groupby('county')['tday_gt90'].mean().max()
counties_zero_days_above_90F = (df_main.groupby('county')['tday_gt90'].mean() == 0).sum()
max_days_above_90F, max_days_value, counties_zero_days_above_90F

#Q2a
total_deaths_over_65 = df_main['deaths'].sum()
total_population_over_65 = df_main['population'].sum()
national_average_mortality_rate_over_65 = (total_deaths_over_65 / total_population_over_65) * 100000
national_average_mortality_rate_over_65
weighted_sum_cruderate = (df_main['cruderate'] * df_main['population']).sum()
national_average_mortality_rate_cruderate = weighted_sum_cruderate / total_population_over_65
national_average_mortality_rate_cruderate
 
 #Q2b
total_deaths = df_main['deaths'].sum()
total_deaths

##############################################

#Q3a

df_crosswalk = pd.read_csv(os.path.join(PATH, fname2))
df_crosswalk.head()
grouped_df = df_main.groupby('countycode').mean().reset_index()
grouped_df = df_main.groupby('countycode').mean()
temp_columns = grouped_df.columns[grouped_df.columns.str.startswith('tday')]
hotdays_columns = [col for col in temp_columns if '70_80' in col or '80_90' in col or 'gt90' in col]
hotterdays_columns = [col for col in temp_columns if '80_90' in col or 'gt90' in col]
grouped_df['hotdays'] = grouped_df[hotdays_columns].sum(axis=1)
grouped_df['hotterdays'] = grouped_df[hotterdays_columns].sum(axis=1)
grouped_df
mean_vars = ['deaths','population', 'cruderate']
mean_data = df_main.groupby(['countycode']).mean()[mean_vars].reset_index()
mean_data.tail()
merged_data = pd.merge(grouped_df, df_crosswalk, on='countycode', how='inner')
merged_data.head()
merged_data.columns
#merged_data = merged_data.merge(df_main[['countycode','hotdays', 'hotterdays']], on=['countycode'], how='inner')
#merged_data.to_excel('doublecheck.xlsx', index=False)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='normal_1981_2010', y='cruderate', data=merged_data)
plt.title('Relationship between County Average Temperatures and Over-65 Mortality Rates')
plt.xlabel('Average Temperature (1981-2010)[°C]')
plt.ylabel('Over-65 Mortality Rate (per 100,000)')
slope, intercept, r_value, p_value, std_err = linregress(merged_data['normal_1981_2010'],
merged_data['cruderate'])
sns.regplot(x='normal_1981_2010', y='cruderate', data=merged_data,
scatter=False, color='red', line_kws={"label":f"y={slope:.2f}x+{intercept:.2f}"})
plt.legend()
plt.show()
slope, r_value

 #Q3b
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='hotdays', y='cruderate', data=merged_data)
plt.title('Mortality Rates vs. Hot Days (over 70°F)')
plt.xlabel('Number of Hot Days (over 70°F)')
plt.ylabel('Over-65 Mortality Rate (per 100,000)')
slope_hotdays, intercept_hotdays, r_value_hotdays, _, _ = linregress(merged_data['hotdays'], merged_data['cruderate'])
sns.regplot(x='hotdays', y='cruderate', data=merged_data, scatter=False, color='red',
line_kws={"label":f"y={slope_hotdays:.2f}x+{intercept_hotdays:.2f}"})
plt.legend()
plt.subplot(1, 2, 2)
sns.scatterplot(x='hotterdays', y='cruderate', data=merged_data)
plt.title('Mortality Rates vs. Hotter Days (over 80°F)')
plt.xlabel('Number of Hotter Days (over 80°F)')
plt.ylabel('Over-65 Mortality Rate (per 100,000)')
slope_hotterdays, intercept_hotterdays, r_value_hotterdays, _, _ = linregress(merged_data['hotterdays'], merged_data['cruderate'])
sns.regplot(x='hotterdays',
y='cruderate', data=merged_data, scatter=False, color='red',
line_kws={"label":f"y={slope_hotterdays:.2f}x+{intercept_hotterdays:.2f}"})
plt.legend()
plt.tight_layout()
plt.show()
(slope_hotdays, r_value_hotdays),
(slope_hotterdays, r_value_hotterdays)

 #Q4a
new_df_main = df_main.copy()
temp_columns = new_df_main.columns[new_df_main.columns.str.startswith('tday')]
hotdays_columns = [col for
col in temp_columns if '70_80' in col or '80_90' in col or 'gt90' in col]
hotterdays_columns = [col for col in temp_columns if '80_90' in col or 'gt90' in col]
new_df_main['hotdays'] = new_df_main[hotdays_columns].sum(axis=1)
new_df_main['hotterdays'] = new_df_main[hotterdays_columns].sum(axis=1)
new_df_main[['county', 'year', 'hotdays', 'hotterdays']].head()
selected_counties = ['Mobile County, AL', 'Cook County, IL', 'Los Angeles County, CA', 'Miami-Dade County, FL']
filtered_df = new_df_main[new_df_main['county'].isin(selected_counties)]
plt.figure(figsize=(10, 6))
sns.regplot(x='hotterdays', y='cruderate', data=filtered_df,
scatter_kws={'alpha':0.5})
plt.title('Death Rates vs Hotter Days (All Selected Counties)')
plt.xlabel('Hotter Days')
plt.ylabel('Crude Death Rate (per 100,000)')
plt.show()

 #Q4b
plt.figure(figsize=(12, 8))
for county in selected_counties:
    county_data = filtered_df[filtered_df['county'] == county]
    sns.regplot(x='hotterdays', y='cruderate', data=county_data, label=county, scatter_kws={'alpha':0.5})
plt.title('Death Rates vs Hotter Days (Each County Individually)')
plt.xlabel('Hotter Days')
plt.ylabel('Crude Death Rate (per 100,000)')
plt.legend()
plt.show()