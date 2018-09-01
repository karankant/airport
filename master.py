# -*- coding: utf-8 -*-
"""
Final Project CS 5010

Alex Gromadzki (arg2eu)
Karan Kant (kk4ze)
Sung Min Yang (sy8pa)

Group Name: Group 8
"""

#import requests               # webscraping kept in separate file
#from bs4 import BeautifulSoup

import pandas #import pandas, matplotlib and seaborn libraies
import matplotlib.pyplot as mpl
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

sns.set(style="darkgrid") #set graphs as dark grid 
Columnlist=['Air Carrier','Air Taxi', 'Military','Local Civil', 'Local Military'] #specify column names that will be included in datframes

region_abbreviations = {"A": "Alaska", "C": "Central",  "E": "Eastern", #abbreviate regions so they are easier to call upon later
                        "GL": "Great Lakes", "NE": "New England", "NW": "Northwest",
                        "SW": "Southwestern", "S": "Southern", "WP": "Western Pacific"}
troubled_regions = ["Alaska","Great Lakes"] #regions that need special read and clean instructions due to csv format

def readandclean(abbreviation, start_data):  # looks for region abbreviation 
                                            # and location where first datapoints start (varies per set)
    abbreviation = abbreviation.upper() #sterility check to ensure no variance in input
    try:
        csv = region_abbreviations[abbreviation]  # extract full region name from dictionary, using uppercase for the key
    except KeyError: #ask for a proper input in case of a different input
        raise(KeyError('The region code you entered is not supported at this time.')) #raise error and print line
        
    if csv not in troubled_regions: #if CSV is not Alaska Great Lakes
        Name=pandas.read_csv(str(csv) +'.csv') #read in CSV into dataframe by pandas
        Name.dropna(inplace = True)             # gets rid of blank lines 
        Name.columns = ['Date','Airport', 'Air Carrier','Air Taxi',
                   'General Aviation', 'Military', 'Total',
                   'Iterative Civil', 'Iterative Military', 'Iterative Total',
                   'Total Operations']         # column headers
        
        Name['Date']=pandas.to_datetime(Name.Date, errors='coerce').dt.date #convert date column to date format
        Name.sort_values(['Date','Airport'],ascending=[True,True]) #sort values by date and airport ascending
        Name['Region']=csv
        Name.drop(['General Aviation','Total','Iterative Total', 'Total Operations'],axis=1,inplace=True) #drop unneeded variables
        Name.columns = ['Date','Airport', 'Air Carrier','Air Taxi', 'Military','Local Civil', 'Local Military','Region'] #define columns
        Name['Air Carrier'] = Name['Air Carrier'].str.replace(',', '') #replace columns in numbers with spaces for each type
        Name['Air Taxi'] = Name['Air Taxi'].str.replace(',', '')
        Name['Local Civil'] = Name['Local Civil'].str.replace(',', '')
        Name['Military'] = Name['Military'].str.replace(',', '')
        Name['Local Military'] = Name['Local Military'].str.replace(',', '')
        Name = Name.iloc[start_data:] #start dataframe from specified row to delete 1989 data

    elif csv == "Alaska"  : #special case for Alaska that does not have major military activity. Only difference is military string
        # commas were not replaced, otherwise is exactly the same workflow as above
        Alaska=pandas.read_csv('Alaska.csv')
        Alaska.dropna(inplace = True) 
        Alaska.columns = ['Date','Airport', 'Air Carrier','Air Taxi',
                   'General Aviation', 'Military', 'Total',
                   'Iterative Civil', 'Iterative Military', 'Iterative Total',
                   'Total Operations']
        Alaska['Date']=pandas.to_datetime(Alaska.Date, errors='coerce').dt.date
        Alaska.sort_values(['Date','Airport'],ascending=[True,True])
        Alaska['Region']='Alaska'
        Alaska.drop(['General Aviation','Total','Iterative Total', 'Total Operations'],axis=1,inplace=True)
        Alaska.columns = ['Date','Airport', 'Air Carrier','Air Taxi', 'Military','Local Civil', 'Local Military','Region']
        Alaska['Air Carrier'] = Alaska['Air Carrier'].str.replace(',', '')
        Alaska['Air Taxi'] = Alaska['Air Taxi'].str.replace(',', '')
        Alaska['Local Civil'] = Alaska['Local Civil'].str.replace(',', '')
        Alaska['Local Military'] = Alaska['Local Military'].str.replace(',', '')
        Alaska = Alaska.iloc[start_data:]
        return Alaska
        
    elif csv == "Great Lakes"   :    #Great Lakes csv had empty columns at the end, unnamed 11 to unnamed 21. Special case to 
        # delete them otherwise is exactly the same workflow as above
        GreatLakes=pandas.read_csv('Great Lakes.csv')
        GreatLakes.drop(['Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15',
                     'Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19','Unnamed: 20','Unnamed: 21'],axis=1,inplace=True)
        GreatLakes.dropna(inplace = True) 
        GreatLakes.columns = ['Date','Airport', 'Air Carrier','Air Taxi',
                   'General Aviation', 'Military', 'Total',
                   'Iterative Civil', 'Iterative Military', 'Iterative Total',
                   'Total Operations']
        GreatLakes['Date']=pandas.to_datetime(GreatLakes.Date, errors='coerce').dt.date
        GreatLakes.sort_values(['Date','Airport'],ascending=[True,True])
        GreatLakes['Region']='Great Lakes'
        GreatLakes.drop(['General Aviation','Total','Iterative Total', 'Total Operations'],axis=1,inplace=True)
        GreatLakes.columns = ['Date','Airport', 'Air Carrier','Air Taxi', 'Military','Local Civil', 'Local Military','Region']
        GreatLakes['Air Carrier'] = GreatLakes['Air Carrier'].str.replace(',', '')
        GreatLakes['Air Taxi'] = GreatLakes['Air Taxi'].str.replace(',', '')
        GreatLakes['Local Civil'] = GreatLakes['Local Civil'].str.replace(',', '')
        GreatLakes['Military'] = GreatLakes['Military'].str.replace(',', '')
        GreatLakes['Local Military'] = GreatLakes['Local Military'].str.replace(',', '')
        GreatLakes = GreatLakes.iloc[start_data:]
        return GreatLakes
        
    return Name #return cleaned dataframe
     
def plotit(name,var,filename=None): #plotit function that will plot a region dataframe to multiple operation types. 
    # name=region dataframe name, var= list of operational types to be plotted, optional filename input if a pdf is required
    plot_all = mpl.figure() #initialize myplotlib canvas
    plot_all.set_size_inches(16, 13.5) #set canvas size
    var_list=str(var).strip("[]")
    mpl.title(name['Region'][0]+':'+str(var_list), fontsize=20) #set title and title size
    for column in var: #for each operational type
        rollingmean='rollingmean '+column #set rolling mean varaible
        sns.set_context("notebook") #plot line width
        mpl.plot_date(name['Date'], name[column],alpha=0.3) #plot data points
        sns.set_context("poster") #thicken line width
        rolling_mean_label = str("Rolling Mean " + column ) #legend label
        sns.lineplot(x='Date', y=str(rollingmean),data=name,label= rolling_mean_label) #plot rolling mean line using seaborn library
        
    mpl.xlabel('Year', fontsize=16) #label x axis
    mpl.ylabel('Change in % relative to 1990 ', fontsize=18) #label y axis

    mpl.legend() #output legend

    if filename is not None: #if a filename is specified:
        savename = (str(filename) + ".pdf") #output pdf
        mpl.savefig(savename) #save figure
    mpl.show() #show graph
 

def ratio(name): #ratio function that normalizes dataframe to a base 100 at the year 1990
    #group by every column with their sum 
    name_sum = name.groupby("Date")['Air Carrier','Air Taxi','Military','Local Civil','Local Military'].apply(lambda x : x.astype(int).sum()).reset_index()
    #divide each column value with their first 1990 value to generate a value dependant on the base 100 rate
    name_sum_ratio=name_sum.loc[:, 'Air Carrier':]= name_sum.loc[:, 'Air Carrier':].div(name_sum.iloc[0]['Air Carrier':]/100)
    name_sum_ratio['Date']=name_sum['Date'] #append date column to dataframe
    name_sum_ratio['Region']=name['Region'][50] #set region 
    #create a rolling mean for each type  that divides the data into 15 chunks and calculates the average over each iteration 
    name_sum_ratio['rollingmean Air Carrier'] = name_sum_ratio['Air Carrier'].rolling(15).mean() 
    name_sum_ratio['rollingmean Air Taxi'] = name_sum_ratio['Air Taxi'].rolling(15).mean()
    name_sum_ratio['rollingmean Military'] = name_sum_ratio['Military'].rolling(15).mean()
    name_sum_ratio['rollingmean Local Civil'] = name_sum_ratio['Local Civil'].rolling(15).mean()
    name_sum_ratio['rollingmean Local Military'] = name_sum_ratio['Local Military'].rolling(15).mean()
    #convert all values to numeric
    name_sum_ratio[Columnlist] = name_sum_ratio[Columnlist].apply(pandas.to_numeric) 
    return name_sum_ratio


def plotairport(airport,var): #plot function for a specific airport
    
    airport=MainFrame.loc[MainFrame['Airport'] == airport].reset_index() #extracts all requested airport data from the master datafram
    airport_sum_ratio=ratio(airport) #create the normalization ratio for the airport
    airport.drop(['index'],axis=1,inplace=True) #drop index
    plotit(airport_sum_ratio,var) #call on plotit function to plot the requested airport

def plotmultiple(airport,var,response,filename=None): #plot with inputs for multiple airports and types along with option to supress data points
    plot_all = mpl.figure() #setup canvas like before
    plot_all.set_size_inches(16, 13.5)
    var_list=str(var).strip("[]")
    airport_list=str(airport).strip("[]")
    mpl.title(str(var_list)+': '+str(airport_list), fontsize=20)
    if response=='Y': #if data points are supressed, we do not need to us mplot for our oplot
        for word in airport: #for every airport in the list
            tempdf=MainFrame.loc[MainFrame['Airport'] == word].reset_index() #create temporary dataframe that will hold airport data
            tempdf_sum_ratio=ratio(tempdf) #create ratio for airport
            for column in var: #for every operation type
                rollingmean='rollingmean '+column #define rolling mean for operation type
                sns.set_context("poster") #set line width
                rolling_mean_label = str(word+" Rolling Mean " + column )
                sns.lineplot(x='Date', y=str(rollingmean),data=tempdf_sum_ratio,label= rolling_mean_label) #print rolling mean line plot 
    if response=='N': #if no data point supression requested:
        for word in airport:#do the same loop as above, except add mpl plot for data points
             tempdf=MainFrame.loc[MainFrame['Airport'] == word].reset_index()
             tempdf_sum_ratio=ratio(tempdf)
             for column in var:
                rollingmean='rollingmean '+column
                sns.set_context("notebook")
                mpl.plot_date(tempdf_sum_ratio['Date'], tempdf_sum_ratio[column],alpha=0.3) #add data points
                sns.set_context("poster")
                rolling_mean_label = str(word+" Rolling Mean " + column )
                sns.lineplot(x='Date', y=str(rollingmean),data=tempdf_sum_ratio,label= rolling_mean_label) 
             
    mpl.xlabel('Year', fontsize=16) #label plots
    mpl.ylabel('Change in % relative to 1990 ', fontsize=16)
    
    if filename is not None: #output pdf if required
        savename = (str(filename) + ".pdf")
        mpl.savefig(savename)
    mpl.show()
    
def plotmultipleaggr(airport,var,response,filename=None): #simliar function as above but plotted aggregate operation amounts, not a ratio
    plot_all = mpl.figure() #initialize canvas
    plot_all.set_size_inches(16, 13.5)
    var_list=str(var).strip("[]")
    airport_list=str(airport).strip("[]")
    mpl.title(str(var_list)+': '+str(airport_list), fontsize=20)
    if response=='Y': #if data points are surpressed
        for word in airport:
            tempdf=MainFrame.loc[MainFrame['Airport'] == word].reset_index() #create temporary dataframe for airport
            tempdf['rollingmean Air Carrier'] = tempdf['Air Carrier'].rolling(15).mean() #create rolling means for temporary dataset
            tempdf['rollingmean Air Taxi'] = tempdf['Air Taxi'].rolling(15).mean()
            tempdf['rollingmean Military'] = tempdf['Military'].rolling(15).mean()
            tempdf['rollingmean Local Civil'] = tempdf['Local Civil'].rolling(15).mean()
            tempdf['rollingmean Local Military'] = tempdf['Local Military'].rolling(15).mean()
            tempdf[Columnlist] = tempdf[Columnlist].apply(pandas.to_numeric)  
            for column in var: #for every operation type
                rollingmean='rollingmean '+column #define rolling mean by type
                sns.set_context("poster")
                rolling_mean_label = str(word+" Rolling Mean " + column )
                sns.lineplot(x='Date', y=str(rollingmean),data=tempdf,label= rolling_mean_label)  #print plot
    if response=='N': #if no data point surpression, do the same as above but with the added data point plots
        for word in airport:
             tempdf=MainFrame.loc[MainFrame['Airport'] == word].reset_index()
             tempdf['rollingmean Air Carrier'] = tempdf['Air Carrier'].rolling(15).mean()
             tempdf['rollingmean Air Taxi'] = tempdf['Air Taxi'].rolling(15).mean()
             tempdf['rollingmean Military'] = tempdf['Military'].rolling(15).mean()
             tempdf['rollingmean Local Civil'] = tempdf['Local Civil'].rolling(15).mean()
             tempdf['rollingmean Local Military'] = tempdf['Local Military'].rolling(15).mean()
             tempdf[Columnlist] = tempdf[Columnlist].apply(pandas.to_numeric)
             for column in var:
                rollingmean='rollingmean '+column
                sns.set_context("notebook")
                mpl.plot_date(tempdf['Date'], tempdf[column],alpha=0.3) #add datapoint plot
                sns.set_context("poster")
                rolling_mean_label = str(word+" Rolling Mean " + column )
                sns.lineplot(x='Date', y=str(rollingmean),data=tempdf,label= rolling_mean_label) 
    mpl.xlabel('Year', fontsize=16)
    mpl.ylabel('Operations', fontsize=16)
    
    if filename is not None:
        savename = (str(filename) + ".pdf")
        mpl.savefig(savename)
    mpl.show()   
       
def userInput(): #EUser input tool that will extract airport and operation type data depending on user request
    #print instructions
    print('Welcome to the Interactive Airport Selection Tool.\nPlease type in the three digit IATA code(s) of the airport(s) in the United States you want to plot.')
    print('Please seperate each airport code with a comma and no spaces. Example: JFK,EWR,LGA')
    airports=input('Input Airports: ') #hold airport inputs
    airportslist=[x.strip() for x in airports.split(',')] #convert airport inputs to list
    airports.split() 
    print('Please input the traffic type(s). Options include: Air Carrier, Air Taxi, Military')
    traffic=input('Input type: ') #record inputs of traffic types
    trafficlist=[x.strip() for x in traffic.split(',')] #split inputs into another list
    selection=int(input('Please input 0 or 1 if you require 0. 1990 Normalized data or 1. Real value data: ')) #input whether 
    #the user wants normalized or real value data
    response=input('Surpress individual data points?(Y/N): ').upper() #input response whether user wants to surpess data points
    file=input('Input Y if you want to create a pdf output, otherwise input N: ').upper() #input whether the user wants to export a PDF output
    file.strip() #strip whitespace
    if selection==0: #if the user wants normalized data, run this
        if file=='Y': #if user wants to output a PDF
            file2=input('Please specify the filename: ') #input filename
            plotmultiple(airportslist,trafficlist,response,file2) #run plot
        elif file=='N':#if the user doesn't want a pdf output, only run plot
            plotmultiple(airportslist,trafficlist,response)
    elif selection==1: #if the user wants real value data, run this
          if file=='Y':
            file2=input('Please specify the filename: ')
            plotmultipleaggr(airportslist,trafficlist,response,file2) #run aggregate function
          elif file=='N':
            plotmultipleaggr(airportslist,trafficlist,response)
            
            
            
if __name__ == "__main__": #initialize main function
    
    Alaska = readandclean('A',6) #initialize read and clean functions to extract clean dataframes for each region
    Central = readandclean('C', 6)
    Eastern = readandclean('E', 34)
    NewEngland = readandclean('NE', 7)
    Northwest = readandclean('NW', 13)
    Southwestern = readandclean('SW', 19)
    Southern = readandclean('S', 37)
    WesternPacific = readandclean('WP', 28)
    GreatLakes = readandclean('GL',28)
    

    MainFrameList=[Alaska, Central, Eastern, GreatLakes, NewEngland, Northwest, Southern, Southwestern, WesternPacific] #stack region
    # dataframes together to create mainframelist
    MainFrame = pandas.concat(MainFrameList) #concatenate all region dataframes together
    MainFrame.sort_values(['Date','Airport'],ascending=[True,True]) #sort data

    MainFrame.to_csv('AirportMainFrame.csv')
    #create normalization ratio dataframes for each region
    Alaska_sum_ratio=ratio(Alaska)
    Central_sum_ratio=ratio(Central)
    Eastern_sum_ratio=ratio(Eastern)
    GreatLakes_sum_ratio=ratio(GreatLakes)
    NewEngland_sum_ratio=ratio(NewEngland)
    Northwest_sum_ratio=ratio(Northwest)
    Southern_sum_ratio=ratio(Southern)
    Southwestern_sum_ratio=ratio(Southwestern)
    WesternPacific_sum_ratio=ratio(WesternPacific)
    
    
    #sample plots below
    plotit(Alaska_sum_ratio,['Air Carrier','Air Taxi']) #testing air carriers vs taxi hypothesis
    plotit(Central_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(Eastern_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(GreatLakes_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(NewEngland_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(Northwest_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(Southern_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(Southwestern_sum_ratio,['Air Carrier','Air Taxi'])
    plotit(WesternPacific_sum_ratio,['Air Carrier','Air Taxi'])
    
    plotairport('MSY',['Military']) #individual airport plots 
    plotairport('MSY',['Air Carrier']) 
    plotairport('IAD',['Air Carrier'])
    plotairport('DCA',['Air Carrier'])
    plotairport('RDU',['Air Carrier'])
    plotmultiple(['JFK','LAX','BOS','ORD','ATL'],['Air Carrier'],'Y') #multiple plots
  
    #create a mainframe ratio dataframe in order to subset specific date values
    MainFrameRatioList=[Alaska_sum_ratio, Central_sum_ratio, Eastern_sum_ratio, GreatLakes_sum_ratio, NewEngland_sum_ratio, Northwest_sum_ratio, Southern_sum_ratio, Southwestern_sum_ratio, WesternPacific_sum_ratio] 
    MainFrame_ratio=pandas.concat(MainFrameRatioList) #change the mainframe into a ratio
    plot_all = mpl.figure() #initialize canvas 
    plot_all.set_size_inches(25, 18) 
    mpl.title('Air Carrior Operations- per region', fontsize=16) 
    sns.lineplot(x="Date", y="rollingmean Air Carrier",
             hue="Region",
             data=MainFrame_ratio)  #plot sns time graph divided by region for air carrier operations
    mpl.show() 
    
    l = [] #create blank array in order to convert datetime format into string
    for d in MainFrame_ratio['Date']:  #for every value in Date column, convert to string format 
        l.append(d.strftime('%Y%d%m')) #NOTE: This will only run once, once the date is already in string format then it will throw an error
    MainFrame_ratio['Date'] = pandas.Series([a for a in l]) #put in column
    
    MainFrame_911A=MainFrame_ratio.loc[MainFrame_ratio['Date'] == '20010108'] #subset dataframes based on months August and September 2001
    MainFrame_911B=MainFrame_ratio.loc[MainFrame_ratio['Date'] == '20010109']
    MainFrame911=pandas.concat([MainFrame_911A,MainFrame_911B]) #concatenate list
    plot_all = mpl.figure() #initialize canvas
    plot_all.set_size_inches(29.6, 25)
    mpl.title('August 2001 vs September 2001 Operations- Air Carrier, by Region', fontsize=25)
    sns.barplot(x="Region", y="Air Carrier",
             hue="Date",
             data=MainFrame911)  #plot bar plot showcasing both months per region
    mpl.show() 
    
    #-------------------------------------#
    
    
    Dulles = pandas.read_csv("Reduced_MWAA_IAD.csv")
    Dulles.columns 
    
    # have to rebase the total passengers to 1990 to compare with air carrier
    for row in range(len(Dulles)):
        Dulles["Total Passengers"][row] = Dulles["Total Passengers"][row]*((100)/Dulles["Total Passengers"][0])
    
    # need to make sure that the date is recognized as a date for old dulles data
        # but these will acutally convert the date to millisecunds
    #Dulles['Year'] = pandas.to_datetime(Dulles['Year'], yearfirst= True)
    #Dulles['Year'] = Dulles['Year'].dt.year
    
    
    iad = MainFrame.loc[MainFrame['Airport'] == 'IAD'].reset_index()
    iad_sum_ratio=ratio(iad)       # sets cumulative percent growth
    iad.drop(['index'],axis=1,inplace=True) # note this is on old iad, not update
    
    iad_sum_ratio['Date'] = pandas.to_datetime(iad_sum_ratio['Date'])
    
    
    iad_sum_ratio['Year'] = iad_sum_ratio['Date'].dt.year
    air_carrier_iad_by_year= iad_sum_ratio.groupby(["Year"])["Air Carrier"].mean().reset_index()
    for row in range(len(air_carrier_iad_by_year)):
        air_carrier_iad_by_year["Air Carrier"][row] = air_carrier_iad_by_year["Air Carrier"][row]*((100)/air_carrier_iad_by_year["Air Carrier"][0])
        # cumulative yearly growth, setting 1990 as a base of 100
    
    
    DullesFinal = pandas.merge(Dulles, air_carrier_iad_by_year, on="Year")
        
    DullesFinal = DullesFinal[['Year','Total Passengers','Air Carrier']]
    DullesFinal.columns
    DullesFinal['Year'] = pandas.to_datetime(DullesFinal['Year'], format='%Y.0')
        
    
    plot_DullesFinal = mpl.figure()
    plot_DullesFinal.set_size_inches(16, 13.5)
    mpl.title("IAD: Cumulative Growth of Total Passengers and Air Carrier Operations", fontsize=23)
    
    y_columns_dulles = ['Total Passengers', 'Air Carrier']
    for column in y_columns_dulles:
        sns.set_context("notebook")
        mpl.plot_date(DullesFinal['Year'], DullesFinal[column],alpha=0.5)
        sns.lineplot(x='Year', y=column,data=DullesFinal,label=str(column))
        sns.set_context("poster")
    plot_DullesFinal.autofmt_xdate()
    mpl.xlabel('Year', fontsize=16)
    mpl.ylabel('Cumulative Change in % ', fontsize=16)
    
    mpl.legend()
    mpl.savefig('MWAA vs FAA.pdf')
    
    # subquestion 1: does adding air taxi on top show anything interesting

    year_carrier_taxi= iad_sum_ratio.groupby(["Year","Air Carrier","Air Taxi"])
    year_carrier_taxi= iad_sum_ratio.groupby(["Year"])["Air Carrier","Air Taxi"].mean().reset_index()

    for row in range(len(year_carrier_taxi)):
        year_carrier_taxi["Air Carrier"][row] = year_carrier_taxi["Air Carrier"][row]*((100)/year_carrier_taxi["Air Carrier"][0])
        # cumulative yearly growth, setting 1990 as a base of 100
        year_carrier_taxi["Air Taxi"][row] = year_carrier_taxi["Air Taxi"][row]*((100)/year_carrier_taxi["Air Taxi"][0])

    

    
    DullesFinal2 = pandas.merge(Dulles, year_carrier_taxi, on="Year")
    
    DullesFinal2 = DullesFinal2[['Year','Total Passengers','Air Carrier', 'Air Taxi']]
    DullesFinal2.columns
    DullesFinal2['Year'] = pandas.to_datetime(DullesFinal2['Year'], format='%Y.0')
        
    
    plot_DullesFinal2 = mpl.figure()
    plot_DullesFinal2.set_size_inches(16, 13.5)
    mpl.title("IAD: Cumulative Growth    \nTotal Passengers, Air Carrier, and Air Taxi Operations", fontsize=23)
    
    y_columns_dulles2 = ['Total Passengers', 'Air Carrier', 'Air Taxi']
    for column in y_columns_dulles2:
        sns.set_context("notebook")
        mpl.plot_date(DullesFinal2['Year'], DullesFinal2[column],alpha=0.5)
        sns.set_context("poster")
        sns.lineplot(x='Year', y=column,data=DullesFinal2,label=str(column))
    plot_DullesFinal2.autofmt_xdate()
    mpl.xlabel('Year', fontsize=16)
    mpl.ylabel('Cumulative Change in % ', fontsize=16)
    
    
    mpl.legend()
    mpl.savefig('Year Passengers Carrier Taxi.pdf')
    
    #-------------------------------------#
#Time Series Decomposition Modelling
    

MainFrame_sum = MainFrame.groupby("Date")['Air Carrier','Air Taxi','Military','Local Civil','Local Military'].apply(lambda x : x.astype(int).sum()) 
#MainFrame_sum_ratio = MainFrame_sum.loc[:, 'Air Carrier':]= jfk_sum.loc[:, 'Air Carrier':].div(jfk_sum.iloc[0]['Air Carrier':]/100)
aircarrier_ts = MainFrame_sum[['Air Carrier']] #So this gives us all operations for US Air Carriers from 1990
airtaxi_ts = MainFrame_sum[['Air Taxi']] #So this gives us all operations for Air Taxis from 1990

# Making sure that the indices are in the required date time format
aircarrier_ts.index = pandas.DatetimeIndex(freq = "m", start = '1989-12-01', periods = 337)

# Making sure that the indices are in the required date time format
airtaxi_ts.index = pandas.DatetimeIndex(freq = "m", start = '1989-12-01', periods = 337)

# The first three observations from 1989 September 30 to November 30 are outliers
# The values are too low and should be exlcluded.
aircarrier_ts = aircarrier_ts['1990-01-31':]
airtaxi_ts = airtaxi_ts['1990-01-31':]


# Decomposing aircarrier into trend, seasonality and residuals
aircarrier_decomposition = seasonal_decompose(aircarrier_ts)
aircarrier_trend = aircarrier_decomposition.trend
aircarrier_seasonal = aircarrier_decomposition.seasonal
aircarrier_residual = aircarrier_decomposition.resid

airtaxi_decomposition = seasonal_decompose(airtaxi_ts)
airtaxi_trend = airtaxi_decomposition.trend
airtaxi_seasonal = airtaxi_decomposition.seasonal
airtaxi_residual = airtaxi_decomposition.resid

# Plots for Air Carrier
# Full US Air Carrier plot
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_ts['2014-12-31':'2015-12-31']/aircarrier_ts.loc['2014-12-31'], label='Full')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier.jpg', bbox_inches='tight')
aircarrier_ts.index = pandas.to_datetime(aircarrier_ts.index)

# US Air Carrier Trend Decomposition
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier Trend', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_trend, label='Trend')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier Trend.jpg', bbox_inches='tight')

#US Air Carrier Seasonality Decomposition
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier Seasonality', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_seasonal[300:],label='Seasonality')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier Seasonality.jpg', bbox_inches='tight')

# Air Carrier Decomposition Residual Plot 
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier Residuals', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_residual, label='Residuals')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier Residuals.jpg', bbox_inches='tight')

# Plots for Air Taxi
# Full Air Taxi Plot
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Taxi', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(airtaxi_ts['2014-04-30':]/airtaxi_ts.loc['2014-04-30'], label='Full')
mpl.legend(loc='best')
mpl.savefig('US Air Taxi.jpg', bbox_inches='tight')

# Air Taxi Trend Decomposition
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Taxi Trend', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(airtaxi_trend, label='Trend')
mpl.legend(loc='best')
mpl.savefig('US Air Taxi Trend.jpg', bbox_inches='tight')

# Air Taxi Seasonality Decomposition
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Taxi Seasonality', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(airtaxi_seasonal[300:],label='Seasonality')
mpl.legend(loc='best')
mpl.savefig('US Air Taxi Seasonality.jpg', bbox_inches='tight')

#Air Taxi Decomposition Residual Plot
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Taxi Residuals', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(airtaxi_residual, label='Residuals')
mpl.legend(loc='best')
mpl.savefig('US Air Taxi Residuals.jpg', bbox_inches='tight')

# Original Airtaxi and Air Carrier
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Taxi vs Air Carrier Base Year 1990', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Number of Operations', fontsize=16)
mpl.plot(aircarrier_ts['1990-01-31':], label='Air Carrier')
mpl.plot(airtaxi_ts['1990-01-31':], label='Air Taxi')
mpl.legend(loc='best')
mpl.savefig('US Air Taxi vs US Air Carrier Base Year 1990.jpg', bbox_inches='tight')

# Air Carrier vs Air Taxi Trend
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier vs Air Taxi Trend', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_trend['1991-01-31':], label='Air Carrier Trend')
mpl.plot(airtaxi_trend['1991-01-31':], label='Air Taxi Trend')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier vs Air Taxi Trend.jpg', bbox_inches='tight')

# Air Carrier vs Air Taxi Seasonality
mpl.figure(figsize=(20, 8.5), dpi=80)
mpl.suptitle('US Air Carrier vs Air Taxi Seasonality', fontsize=16)
mpl.xlabel('Date', fontsize=16)
mpl.ylabel('Operations', fontsize=16)
mpl.plot(aircarrier_seasonal['2016-12-31':'2017-12-31'], label='Air Carrier Seasonality')
mpl.plot(airtaxi_seasonal['2016-12-31':'2017-12-31'], label='Air Taxi Seasonality')
mpl.legend(loc='best')
mpl.savefig('US Air Carrier vs Air Taxi Seasonality.jpg', bbox_inches='tight')



    