"""
Final CS108 Data Analytics Project
Name: Chase Lenhart
Email: chaselen@bu.edu
Assignment: CS108 Final Project (cs108_final.py)

Project Description: The question I am looking to answer is if the release of 
certain ratings of movies or TV shows (G for kids or R for adults for example) 
affects Netflix stock prices, indicating what types of ratings should Netflix 
release. To start, I will be doing an analysis of the stock price dataframe, and,
with this dataframe, plotting, finding probabilities, etc. I will use two data sets
for the project: Netflix Yahoo Finance Stock prices and a database from kaggle 
(https://www.kaggle.com/datasets/imtkaggleteam/netflix) which includes data on the 
ratings and release dates of over 7000 Netflix movies or TV shows.

CS108 Section A1
"""

# =============================================================================
# Import Statements
# =============================================================================
import pandas as pd # pandas for DataFrames and DataSeries
import matplotlib.pyplot as plt # Matplotlib for plotting graphs and statistics
import statsmodels.api as sm # Statsmodels for regression
import yfinance as yahooFinance # Yahoo Finance for Netflix stock price data
from cs108_final_helper_functions import * # helper functions made to help operations in this main file

    
# =============================================================================
# Finding Probabilities in Dataframes
# =============================================================================

def probabilities(df_combo):
    '''
    Function that finds the following probabilities: ratings that positive, ratings
    that are negative, stock price changes that are positive, stock price changes 
    that are negative, the conditional probability that the rating rises given
    the stock rises, and the conditional probability that the stock rises given
    the rating rises. The probabilities are printed
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        None
    '''
    
    # Percentage of rating changes that are positive or negative
    rating_chg_list = df_combo['Rating Count'].to_list()
    rating_pct_positive = pct_positive_in_list(rating_chg_list)
    rating_pct_negative = 1 - rating_pct_positive


    # Altering the df_data_dup (combination df) 
    df_combo['High_chg'] = df_combo['High'] / df_combo['High'].shift(1) - 1 
    df_combo['Rating_chg'] = df_combo['Rating Count'] / df_combo['Rating Count'].shift(1) - 1 


    # Find rows in which both stock rise and rating rise
    df_rows_stock_rating = (df_combo['High_chg'] > 0) & (df_combo['Rating Count'] > 0)
    df_combo_2 = df_combo[df_rows_stock_rating]

    # probability that both stock and rating changes were positive
    both_stock_rating = len(df_combo_2) / len(df_combo)


    # find the percentage of stock changes that are positive and negative
    df_rows_stock_positive = (df_combo['High_chg'] > 0)
    df_rows_stock_positive = df_combo[df_rows_stock_positive]
    stock_postitive_pct = len(df_rows_stock_positive) / len(df_combo)
    stock_negative_pct = 1 - stock_postitive_pct


    print('Probability of Stock Price Rising = ' + str(stock_postitive_pct))
    print('Probability of Stock Price Falling = ' + str(stock_negative_pct))
    
    print('Probability of Rating being Positive = ' + str(rating_pct_positive))
    print('Probability of Rating being Negative = ' + str(rating_pct_negative))

    # conditional probability that given rating went up, what is the probability that stock prices rose
    cond_prob_rating_given_stock = both_stock_rating / stock_postitive_pct

    # find the conditional probability of given rating goes up, what is the prob that stocks rose
    cond_prob_stock_given_rating = both_stock_rating / rating_pct_positive

    print('Probability of Rating Rises Given Stock Rises = ' + str(cond_prob_rating_given_stock))
    print('Probability of Stock Rises Given Rating Rises = ' + str(cond_prob_stock_given_rating))
    

def best_director(df_combo):
    '''
    Function that finds the best director according to these conditions: the director
    in which the stock price of the Netflix stock changes the most (positively) 
    is declared the best director, indicating people like the director the most.
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        max_director - the claimed best director
    '''
    
    # make a duplicate of combo df
    df_dup = df_data
    
    # remove all duplicate dates in df_dup
    duplicate_dates = df_dup[df_dup.index.duplicated()].index.to_list()
    
    # Remove all dates that are duplicates
    df_unique_dates = df_dup[~df_dup.index.isin(duplicate_dates)]
    
    # Add the stock changes to new unique date df
    df_unique_dates['High_chg'] = df_stock['High_chg']

    # preset conditions
    max_price = -9999999
    max_director = ''

    # find all unique director values
    unique_directors = df_unique_dates['director'].unique()

    # loop through directors
    for director in unique_directors:

        # for each director find all stock changes
        df_director_only = df_unique_dates['director'] == director
        df_temp = df_unique_dates[df_director_only]
        
        # ensuring the row is not NaN
        if type(director) != float:
        
            # if the stock change is higher than temp stock change then replace 
            if df_unique_dates['High_chg'].sum() > max_price:
                max_price = df_unique_dates['High_chg'].sum()
                max_director = director
    
    # print results
    print(f"Best Director = {max_director}")
    print(f"Stock Changes for {max_director} = ${max_price}")
    
    # return the best director according to stock rise in prices
    return max_director


# =============================================================================
#  Stock Data Plotting
# =============================================================================

def plot_stock_prices(df_stock):
    '''
    Function that plots the line graph of stock prices for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['High'].plot(color='red', ylabel='Stock Prices', title='Netflix Stock Price Graph')


def plot_stock_returns(df_stock):
    '''
    Function that plots the line graph of stock returns for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['Stock_Return'].plot(color='red', ylabel='Stock Returns', title='Netflix Stock Returns Graph')
    
    
def plot_histogram_stock_prices(df_stock):
    '''
    Function that plots the histogram graph of stock prices for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['High'].hist(color='red', legend = True)
    

def plot_histogram_stock_returns(df_stock):
    '''
    Function that plots the histogram graph of stock returns for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['Stock_Return'].hist(color='red', legend = True)


def plot_kde_stock_prices(df_stock):
    '''
    Function that plots the KDE graph of stock prices for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['High'].plot.kde(color='red')
    

def plot_hexbin_stock_prices_on_volume(df_stock):
    '''
    Function that plots the hexbin plot of stock prices compared to volume for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock.plot(kind='hexbin', gridsize=20, x='Volume', y='High', cmap='Reds', title='Stock Prices on Volume Hexbin Plot', xlabel = 'Volume', ylabel='Stock Price')
    
    
def regression_rm_to_price(df_stock):
    '''
    Function that plots the regression plot of stock prices compared to rolling 
    mean of stock price for Netflix
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_stock['High_rm_pct'] = df_stock['High_rm'] / df_stock['High_rm'].max() * 100
    df_stock = df_stock.dropna(subset=['High_rm_pct'])

    independent = 'High_rm'
    dependent = 'High'

    # make scatter plot and plot it
    plt.scatter(df_stock[independent], df_stock[dependent], color='black')
    plt.xlabel(independent)
    plt.ylabel(dependent)


    # make variables for X and Y
    X = df_stock[independent]
    Y = df_stock[dependent]

    # make a model and plot it
    model = sm.OLS(Y,X).fit()
    print(model.summary())
    # plot the predictions as a line plot
    P = model.predict()
    plt.plot(X, P, color='red')
    
    plt.title('Netflix Rolling Mean to Stock Price Scatter Plot')

    plt.show() # show the plot


# =============================================================================
#  Movie/TV Rating Data Plotting
# =============================================================================

def plot_histogram_ratings(df_data):
    '''
    Function that plots the histogram plot of Netflix tv/movie ratings
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_data['rating'].hist(color='red', legend=True)
    

def plot_kde_ratings(df_data):
    '''
    Function that plots the KDE plot of Netflix tv/movie ratings
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_data['Rating_Cum'].plot.kde(color='red')
    


# =============================================================================
#  Plotting Combo Dataframe (combination of stock prices and ratings)
# =============================================================================

def plot_combo(df_combo):
    '''
    Function that plots the line graph of Netflix tv/movie ratings and Netflix stock prices
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_combo[['High_pct', 'Rating_Cum_pct']].plot(ylabel = 'Percentage', title='High Stock Price Percentage vs. Rating Cumulative Percentage Line Graph')
    
    
def plot_stock_return_to_ratings(df_combo):
    '''
    Function that plots the line graph of Netflix tv/movie ratings and Netflix stock returns 
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    df_combo[['Stock_Return_pct', 'Rating_Cum_pct']].plot()


def plot_combo_scatter(df_combo):
    '''
    Function that plots the scatter plot of Netflix tv/movie ratings and Netflix stock prices
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    independent = 'Rating_Cum_pct'
    dependent = 'High_pct'

    # make scatter plot and plot it
    plt.scatter(df_combo[independent], df_combo[dependent], color='black')
    plt.xlabel(independent)
    plt.ylabel(dependent)


    # make variables for X and Y
    X = df_combo[independent]
    Y = df_combo[dependent]

    # make a model and plot it
    model = sm.OLS(Y,X).fit()
    print(model.summary())
    # plot the predictions as a line plot
    P = model.predict()
    plt.plot(X, P, color='red')
    plt.title('Netflix Rating Cumulative Percentage to Netflix Stock Price Percentage Scatter Plot')
    plt.show() # show the plot
    

def plot_combo_kde(df_combo):
    '''
    Function that plots the KDE graph of Netflix tv/movie ratings and Netflix stock prices
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_combo[['High_pct', 'Rating_Cum_pct']].plot.kde()
    

def plot_combo_hexbin(df_combo, size=20):
    '''
    Function that plots the hexbin plot of Netflix tv/movie ratings and Netflix stock prices
    
    Parameters:
        df_combo - the dataframe passed
    
    Return:
        none
    '''
    
    df_combo.plot(kind='hexbin', xlabel='Rating_Cum_pct', x='Rating_Cum_pct', y='High_pct', cmap='Reds', gridsize=size, title='Stock Price Percentage on Rating Cumulative Percentage')








# =============================================================================
# Test Code
# =============================================================================
if __name__ == '__main__':
    
   # Link to Kaggle Dataset - https://www.kaggle.com/datasets/imtkaggleteam/netflix/
   dataset_filename = 'NetFlix.csv'

   # Make variable to hold Netflix stock data
   netflixStock = yahooFinance.Ticker('NFLX')
   
   # Make dataframes for both datasets
   df_data = pd.read_csv(dataset_filename) # Kaggle Dataset dataframe
   df_stock = netflixStock.history(period = 'max') # Netflix stock price dataframe
   
   
   # =============================================================================
   #  Setting Up Columns and Indices in Both Dataframes
   # =============================================================================
   
   # Indexing the stock dataframe to only include data only in the dataset dataframe time span
   df_stock = df_stock.loc['2008-01-01 00:00:00-05:00': '2021-01-15 00:00:00-05:00']

   # rename the dataframes to have the same index formatting (YYYY-MM-DD)
   df_stock = df_stock.rename(index = lambda s: str(s)[:10])
   df_data.index = pd.to_datetime(df_data['date_added'])

   # remove any rows in dataset dataframe that do not have a date
   df_data = df_data.drop(df_data[df_data['date_added'] == '0000-00-00'].index)

   # Sort both dataframes in ascending order of their indices (dates)
   df_data = df_data.sort_index(axis = 0, ascending = True)
   df_stock = df_stock.sort_index(axis = 0, ascending = True)

   # remove all of the unneccesary columns in the dataframes
   '''
   High - the highest price of the stock during the day
   Low - the lowest price of the stock during the day
   Volume - the number of shares traded on the day
   Open - the open stock price on this day
   Close - the close stock price on this day
   '''
   stock_columns = ['High', 'Low', 'Volume', 'Open', 'Close']

   '''
   Title - title of the TV or movie
   Type - Is the release a TV or movie
   Director - director of TV or show
   Date_added - the date the TV or show was added to Netflix
   Rating - the rating of the TV or show
   Duration - the length of the TV or show
   '''
   data_columns = ['title', 'type', 'director', 'date_added', 'rating', 'duration']
   
   # only use the columns in stock and data column lists
   df_stock = df_stock[stock_columns]
   df_data = df_data[data_columns]
   
   
   
   # Make new columns in stock dataframe:
   # Make a High_chg column which is the change in the High column from day to day
   df_stock['High_chg'] = df_stock['High'] / df_stock['High'].shift(1) - 1 

   # Make a rolling mean column for 30 days
   df_stock['High_rm'] = df_stock['High'].rolling(30).mean()

   # Make a rolling mean percent column
   df_stock['High_rm_pct'] = df_stock['High_rm'] / df_stock['High_rm'].max() * 100
 
   # Make a High pct column which calculates the percent of the max stock price the current stock price is
   df_stock['High_pct'] = df_stock['High'] / df_stock['High'].max() * 100

   # Make a stock return column which calculates the return on the stock day to day
   df_stock['Stock_Return'] = df_stock['Close'].pct_change()

   # Make a stock return pct column which calculates the percent of the max stock return the current stock return is
   df_stock['Stock_Return_pct'] = df_stock['Stock_Return'] / df_stock['Stock_Return'].max() * 100
   
   
   # Make new columns in rating dataframe:
   # Lists that classify what rating is good or bad
   good_movies = ['TV-PG', 'TV-14', 'PG', 'TV-G', 'TV-Y']
   bad_movies = ['TV-MA', 'NR', 'R']

   # Assign movies or TV shows with good ratings with 1 and bad ratings with -1
   df_data['Rating_Count'] = df_data.apply(rating, axis = 1)

   # Find accumulation of ratings using the 1 and -1 system
   df_data['Rating_Cum'] = df_data['Rating_Count'].cumsum()

   # Find and make the percent of the highest cumulated sum for the current cum sum
   df_data['Rating_Cum_pct'] = df_data['Rating_Cum'] / df_data['Rating_Cum'].max() * 100


   # Find the change in the ratings from release to release
   df_data['Rating_chg'] = df_data['Rating_Count'] / df_data['Rating_Count'].shift(1) - 1 
   
   
   
   
   # =============================================================================
   #  Making a Combo Dataframe with both Stock Prices and Ratings Data
   # =============================================================================
   
   # Make a duplicate df of data df
   df_combo = df_data

   # Find the list of the dates that have duplicates
   duplicate_dates = df_combo[df_combo.index.duplicated()].index.to_list()

   # list to hold dates that have already been done (efficiency check)
   dates_done = []


   # go through each duplicate date
   for date in duplicate_dates:
       
       # ensure not in done list
       if date not in dates_done:
       
           # Find the rows that have the same date
           df_all_rows = df_combo.index == date
       
           # make a new df with only the rows with this duplicate date
           df_temp = df_combo[df_all_rows]
           
           # remove all of the dates 
           df_combo.drop(df_combo[df_combo.index == date].index)
           # make a new row with the sum of the rating counts
           new_row = {'date_added': date, 'Rating Count': df_temp['Rating_Count'].sum()}
           # add this enew row to the df
           df_combo = pd.concat([df_combo, pd.DataFrame([new_row])])
           
           dates_done.append(date)
           
   # Make the index of df combo to the date
   df_combo.index = df_combo['date_added']
   df_combo.index = pd.to_datetime(df_combo['date_added'])

   # Alter df to use in graphing
   df_combo = df_combo.sort_index(axis = 0, ascending = True)

   # drop any na rows
   df_combo = df_combo.dropna(subset=['date_added'])
   df_combo = df_combo.dropna(subset=['Rating Count'])

   # make new index with date time format
   df_stock.index = pd.to_datetime(df_stock.index)



   # Add all stock data into this df combo dataframe
   df_combo['High'] = df_stock['High']
   df_combo = df_combo.dropna(subset=['High'])

   df_combo['Rating_Cum'] = df_combo['Rating Count'].cumsum()
   df_combo['Rating_Cum_pct'] = df_combo['Rating_Cum'] / df_combo['Rating_Cum'].max() * 100
   df_combo['High_pct'] = df_stock['High'] / df_stock['High'].max() * 100
   df_combo['High_chg'] = df_stock['High_chg']

   df_combo['Rating_chg'] = df_combo['Rating Count'] / df_combo['Rating Count'].shift(1) - 1 
   
   df_combo['Stock_Return'] = df_stock['Close'].pct_change()
   df_combo['Stock_Return_pct'] = df_combo['Stock_Return'] / df_combo['Stock_Return'].max() * 100
