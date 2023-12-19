"""
CS108 Final Helper Functions
Name: Chase Lenhart
Email: chaselen@bu.edu
Assignment: CS108 Final Project (cs108_final_helper_functions.py)

Description: This file contains the helper functions for the CS108 final main file.
"""

def convert_date(date):
    '''
    Function that takes a date in the format DD-MMM-YY to YYYY-MM-DD. This is used for
    the Kaggle dataframe dates and will convert to the nicely formatted stock price
    dataframe date formatting.
    
    Parameters:
        date - the Kaggle formatted date
    
    Return:
        new_date - the newly formatted date in stock price dataframe format
    '''
    
    if type(date) != float:
        if ',' not in date:    
            date = date.split('-')
            
            day = date[0]
            
            if date[1] == 'Jan':
                month = '01'
            elif date[1] == 'Feb':
                month = '02'
            elif date[1] == 'Mar':
                month = '03'
            elif date[1] == 'Apr':
                month = '04'
            elif date[1] == 'May':
                month = '05'
            elif date[1] == 'Jun':
                month = '06'
            elif date[1] == 'Jul':
                month = '07'
            elif date[1] == 'Aug':
                month = '08'
            elif date[1] == 'Sep':
                month = '09'
            elif date[1] == 'Oct':
                month = '10'
            elif date[1] == 'Nov':
                month = '11'
            else:
                month = '12'
            
            year = '20' + date[2]
            
            new_date = f"{year}-{month}-{day:02}"
            
            return new_date
        else:
            date = date.replace(',','')
            date = date.split(' ')
            
            day = date[1]
            year = date[2]
            
            if date[0] == 'January':
                month = '01'
            elif date[0] == 'February':
                month = '02'
            elif date[0] == 'March':
                month = '03'
            elif date[0] == 'April':
                month = '04'
            elif date[0] == 'May':
                month = '05'
            elif date[0] == 'June':
                month = '06'
            elif date[0] == 'July':
                month = '07'
            elif date[0] == 'August':
                month = '08'
            elif date[0] == 'September':
                month = '09'
            elif date[0] == 'October':
                month = '10'
            elif date[0] == 'November':
                month = '11'
            else:
                month = '12'
                
            new_date = f"{year}-{month}-{day:02}"
            
            return new_date
    
    else: 
        return '0000-00-00' 
    
def rating(row):
    '''
    Function that returns the rating score for the row passed as a parameters
    
    Parameters:
        row - the row (movie or tv show) passed 
    
    Return:
        score - the score (1, -1, 0) given to the row for the rating of the movie or tv show
    '''
    
    # rating criteria
    good_movies = ['TV-PG', 'G', 'TV-Y', 'PG-13', 'TV-14', 'TV-Y7', 'TV-G']
    bad_movies = ['R', 'NR', 'TV-MA']
    
    # if rating is in good movies assign score to 1
    if row['rating'] in good_movies:
        score = 1
    # if rating is in bad movies assign score to -1
    elif row['rating'] in bad_movies:
        score = -1
    # else assign score to 0
    else:
        score = 0
    
    # return the score variable
    return score
    
def pct_positive_in_list(list_dup):
    '''
    Function that finds the percent of positive numbers in a list
    
    Parameters:
        list_dup - the list we are finding the number of positive numbers in 
    
    Return:
        pct - the percent of positive numbers in the list
    '''
    
    # start accumulator at 0
    count = 0
    
    # iterate through list 
    for item in list_dup:
        # if item is positive then add 1 to count
        if item>0:
            count += 1
    
    # find percentage
    pct = count / len(list_dup)
    
    # return percentage
    return pct
