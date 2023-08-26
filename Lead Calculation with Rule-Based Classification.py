#############################################
# Lead Calculation with Rule-Based Classification
#############################################

#############################################
# Business Problem
#############################################
# A game company wants to create level-based new customer definitions (personas) by using some features of its customers,
# and to create segments according to these new customer definitions and to estimate how much the new customers can earn
# on average according to these segments.

# For example: It is desired to determine how much a 25-year-old male user from USA who is an IOS user can earn on average.

#############################################
# Story of Dataset
#############################################
# The Persona.csv dataset contains the prices of the products sold by an international game company and some demographic
# information of the users who buy these products. The data set consists of records created in each sales transaction.
# This means that the table is not deduplicated. In other words, a user with certain demographic characteristics may have
# made more than one purchase.

# Columns
# Price: Customer spend amount
# Source: The type of device the customer is connected with
# Sex: Gender of the customer
# Country: Country of the customer
# Age: Age of the customer

################# Before Application #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

#######################################################
## FIRST VERSION
#######################################################

# general information about the dataset in the persona.csv file

import pandas as pd
df = pd.read_csv("persona.csv")
print(df.head())
print(df.info())
print(df.shape)

# unique source values and frequencies
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# how many unique prices
df["PRICE"].nunique()

# information on how many sales were made at which price
df["PRICE"].value_counts()

# information on how many sales are from which country
df["COUNTRY"].value_counts()

# information on how much is earned in total from sales by country
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# number of sales by source types
df.groupby("SOURCE").agg({"PRICE": "count"})

# price averages by country
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# price averages according to sources
df.groupby("SOURCE").agg({"PRICE": "mean"})

# price averages in country-source breakdown
df.groupby(by= ["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# average earnings in country, source, sex, age breakdown
new_df = df.groupby(by= ["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

# new output sorted by price
agg_df = new_df.sort_values("PRICE", ascending= False)

# converting index names to variable names
# all variables except price in agg_df are index names
agg_df.reset_index(inplace=True)

# converting age variable to categorical variable and adding it to agg_df
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

# defining new level based customers and adding them as variables to the dataset
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)

# after creating customers_level_based values with list comp, these values need to be deduplicated
# can be of more than one expression, for example: USA_ANDROID_MALE_0_18
# it is necessary to take them to groupby and get the price average
agg_df["customers_level_based"].value_counts()

# for this reason, after groupby is done according to the segments, the price averages should be taken and the segments
# should be deduplicated.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

# it is in the customers_level_based index. the process of converting them to variables
agg_df = agg_df.reset_index()
agg_df.head()

# is controlled. each persona is expected to have 1
agg_df["customers_level_based"].value_counts()
agg_df.head()

# segmentation by price
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
agg_df.head()

# estimation of how much revenue new customers can be classified and generate
# examples
# which segment does a 33-year-old Turkish woman using android belong to and how much income is expected to earn on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

# in which segment and on average how much income would a 35-year-old French woman using iOS expect to earn?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]


#######################################################
## SECOND VERSION
#######################################################

import pandas as pd
import matplotlib.pyplot as plt

def segment_customers(df):
        df["customers_level_based"] = df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(),
                                                                                      axis=1)
        df = df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()
        df["SEGMENT"] = pd.qcut(df["PRICE"], 4, labels=["D", "C", "B", "A"])
        return df

def plot_segmentation(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["SEGMENT"].value_counts().index, df["SEGMENT"].value_counts().values)
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.title("Customer Segmentation")
    plt.show()

def main():
    df = pd.read_csv("persona.csv")

    new_df = df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
    agg_df = new_df.sort_values("PRICE", ascending=False)
    agg_df.reset_index(inplace=True)

    bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
    mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
    agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)

    segmented_data = segment_customers(agg_df)
    plot_segmentation(segmented_data)

    new_user_1 = "TUR_ANDROID_FEMALE_31_40"
    new_user_2 = "FRA_IOS_FEMALE_31_40"
    user_1_estimate = segmented_data[segmented_data["customers_level_based"] == new_user_1]["PRICE"].values[0]
    user_2_estimate = segmented_data[segmented_data["customers_level_based"] == new_user_2]["PRICE"].values[0]

    print(f"Estimated income for {new_user_1}: ${user_1_estimate:.2f}")
    print(f"Estimated income for {new_user_2}: ${user_2_estimate:.2f}")

main()
