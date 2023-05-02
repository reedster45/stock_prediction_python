import requests
import datetime
import time
from random import randint
from bs4 import BeautifulSoup

import pandas as pd
from yahoo_fin import stock_info
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# return timestamp of given date string
def timeconverter(datestring):
    YYYY = datestring[:4]
    MM = datestring[5:7]
    DD = datestring[8:]
    current = datetime.datetime(int(YYYY), int(MM), int(DD))
    return int(time.mktime(current.timetuple()))

# get timestamp of previous day
def previous_day(timestp):
    return int(timestp - 86400)

# returns converted timestamp in yyy-mm-dd format
def timestampconverter(timestp):
    return datetime.datetime.fromtimestamp(int(timestp)).strftime('%Y-%m-%d')




# scrapes data from given reddit channel based on enddate, startdate, sort, count and prints it to an excel file in date and string columns
def scrape_channel_history(channel="worldnews", enddate="2023-01-01", startdate="2000-01-01", sort="top", filename="data/worldnews.xlsx"):
    
    # setup url and params for desired query
    # https://socialgrep.com/search?query=before:2023-01-03,after:2023-01-02,/r/worldnews&order_by=top
    url = "https://socialgrep.com/search?query=before:" + enddate + ",after:" + startdate + ",/r/" + channel + "&order_by=" + sort

    # start point must be before end point
    endpoint = timeconverter(enddate)
    startpoint = timeconverter(startdate)

    # start at endpoint (current day) and move towards startpoint (in the past before endpoint)
    current = endpoint

    # iterate through everyday in the time period, scraping top 25 entires every day
    data = []
    while current >= startpoint:
        try:
            # calculate the day before the current and then convert both to normal dates
            day_before_current = previous_day(current)
            day_before_current = timestampconverter(day_before_current)
            current = timestampconverter(current)
            
            # load html data from channel and parse headers
            url = "https://socialgrep.com/search?query=before:" + current + ",after:" + day_before_current + ",/r/" + channel + "&order_by=" + sort
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            h5 = soup.find_all('h5')

            # store results in data list
            row = [current]
            for header in h5:
                row.append(header.find('a').text)
            data.append(row)

        except:
            # too many request, wait and try again
            print("error: too many requests I think")
            time.sleep(randint(3, 5))
            continue

        # succesfully scraped one full day, wait (request overload) and move on to the next day 
        print("Done: ", current)
        current = timeconverter(current)
        current = previous_day(current)
        time.sleep(randint(2, 4))
    
    # write results to excel file
    df = pd.DataFrame(data)
    df.columns = ['date', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    print(df)
    df.to_excel(filename, index=False)


# scrape_channel_history()


# takes data from yahoo_fin for given TICK and prints data to excel
def process_data(TICK):
    # get TICK data from yahoo_fin
    data = stock_info.get_data(TICK, start_date='01/25/2008', end_date='01/01/2023', index_as_date=False, interval='1d')

    # for i in data.index:
    #     date = data['date'].dt.strftime('%Y-%m-%d')
    #     data['date'][i] = date
    data['DATE'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # keeep date as index but also add date colun
    x = data
    x.pop('date')
    dates = x.pop('DATE')
    x.insert(0, 'date', dates)

    y = x.pop('close')
    close_labels = pd.DataFrame()
    close_labels.insert(0, 'date', dates)
    close_labels.insert(1, 'close', y)
    

    y2 = x.pop('adjclose')
    adjclose_labels = pd.DataFrame()
    adjclose_labels.insert(0, 'date', dates)
    adjclose_labels.insert(1, 'adjclose', y2)

    x.to_excel('data/data.xlsx', index=False)
    close_labels.to_excel('data/close.xlsx', index=False)
    adjclose_labels.to_excel('data/adjclose.xlsx', index=False)

# process_data("SPY")



# process more reddit data into excel file
def combine_2008_to_2016():
    data = pd.read_csv('data/reddit_2008_to_2016.csv')
    data.drop(columns=['time_created', 'down_votes', 'over_18', 'author', 'subreddit'], axis=1, inplace=True)
    print(data)

    sorted_data = data.sort_values(['date_created', 'up_votes'], ascending=[True, False])
    # print(sorted_data)

    grouped_data = sorted_data.groupby('date_created', as_index=False).agg(list)
    print(grouped_data)

    all_rows = []
    for i in grouped_data.index:
        row = [grouped_data['date_created'][i]]
        row += grouped_data['title'][i]
        all_rows.append(row)

    # write results to excel file
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={0: 'date'})
    print(df)
    df.to_excel('data/2008.xlsx', index=False)

# combine_2008_to_2016()


# merge all /r/worldnews posts into single excel file
def merge_worldnews():
    old = pd.read_excel('data/2008.xlsx', usecols='A:Z')
    new = pd.read_excel('data/worldnews.xlsx', usecols='A:Z')
    new.drop(new.index[4736:], axis=0, inplace=True)
    old.drop(old.index[705:], axis=0, inplace=True)
    old = old[::-1].reset_index(drop=True)
    data = pd.concat([new, old], ignore_index=True)

    # write to excel
    data.to_excel('data/reddit_data.xlsx', index=False)

# merge_worldnews()


# calculate average sentiment for each day
def calculate_sentiment():
    reddit = pd.read_excel('data/reddit_data.xlsx')
    sentiment = SentimentIntensityAnalyzer()
    print(reddit)
    reddit = reddit.values.tolist()
    # print(reddit)

    lis2D = []
    for row in reddit:
        lis = [row[0]]
        num_cols = 0

        neg = 0
        neu = 0
        pos = 0
        compound = 0

        for col in row:
            if pd.notnull(col):
                num_cols += 1
                vadr = sentiment.polarity_scores(col)

                neg += vadr['neg']
                neu += vadr['neu']
                pos += vadr['pos']
                compound += vadr['compound']

        lis.append(neg / num_cols)
        lis.append(neu / num_cols)
        lis.append(pos / num_cols)
        lis.append(compound / num_cols)
        lis2D.append(lis)
    
    df = pd.DataFrame(lis2D)
    df.columns = ['date', 'neg', 'neu', 'pos', 'compound']
    print(df)
    df.to_excel('data/sentiment_data.xlsx', index=False)

# calculate_sentiment()


# merge reddit sentiment data and stock data for final data
def merge_stock_sentiment():
    features = pd.read_excel('data/feature_data.xlsx')
    labels = pd.read_excel('data/close.xlsx')
    

    print(features)
    print(labels)

    df = pd.merge(labels, features)
    print(df)

    df.to_excel('data/complete_data.xlsx', index=False)

merge_stock_sentiment()





# 1 Polish government politicians condemn “disgrace” as Black Eyed Peas wear rainbow armbands during state TV concert
# 2 Lula da Silva sworn in as Brazil's president, amid fears of violence from Bolsonaro supporters
# 3 Japan joins U.N. Security Council as new nonpermanent member
# 4 Defying Expectations, EU Carbon Emissions Drop To 30-Year Lows
# 5 Taiwan president offers China help to deal with COVID surge
# 6 New Omicron super variant XBB.1.5 detected in India
# 7 Morocco bans China arrivals as concerns grow over Covid-19 surge
# 8 Croatia welcomes 2023 by joining Eurozone and Schengen area
# 9 ‘Hope, joy, euphoria’: Brazilians take to streets to celebrate new era under Lula | Brazil
# 10 German intelligence sees growing activity by Russian secret services
# 11 Ukraine must get long-term support, warns Nato chief
# 13 Dozens of New Year attacks on Berlin firefighters: Firefighters in the German capital, Berlin, have reported multiple attacks while trying to do their job. Across the country
# , there were reports of violence on a "terrifying" New Year for many emergency staff
# 14 China appoints 'wolf warrior' as new foreign minister
# 15 France says 690 cars torched on NYE
# 16 1 dead as Iran forces fire on crowd at slain protesters’ memorial, says rights group
# 17 Extinction Rebellion UK to halt disruptive protests
# 18 South Korea Population Declines for 36th Month due to Low Births, Aging Society
# 19 /r/WorldNews Live Thread: Russian Invasion of Ukraine Day 312, Part 1 (Thread #453)
# 20 Outgoing far-right Brazilian President Jair Bolsonaro arrives in Florida, VP Hamilton Mourao is now acting president of Brazil
# 21 Australian government announces COVID-19 testing requirements for travellers from China
# 22 Avianca Changes Rules After Carrying 25 Service Dogs In 1 Flight
# 23 China accuses US of ‘slander, hype’ after aircraft clash
# 24 Iranian police detain top footballers at New Year’s Eve party
# 25 One of Most Crowded Cities, Dhaka, Bangladesh, Gets First Mass-Transit Rail - Bloomberg