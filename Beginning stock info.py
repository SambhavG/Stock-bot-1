import yfinance as yf
import csv


g = open('sp500tickers.csv')
reader = csv.reader(g)
f = open('Stock data.csv', 'a')
writer = csv.writer(f)

for row in reader:
    
    
    ticker = row[0]
    print("Writing data for " + ticker)
    yfTicker = yf.Ticker(ticker)
    
    #Get stock info for month of Jan 2015; first 15 entries
    #y label is 1 if last day of feb. price is above last day of jan. price
        #and false otherwise
    for j in range(2010, 2020):
        hist = yfTicker.history(start=str(j)+'-01-01', end=str(j)+'-02-28',prepost=True)
        if (hist.shape[0] >= 16):
            rowData = []
            for i in range(0, 15):
                closePrice = hist.iloc[i,3]
                rowData.append(closePrice)

            lastClosePrice = hist.iloc[hist.shape[0]-1,3]
            rowData.append(lastClosePrice)

            #rowScaled = list(map(lambda x:translate(x,min(rowData[0:15]),max(rowData[0:15]),0,1), row))
            rowScaled = []
            minMonth = min(rowData[0:15])
            maxMonth = max(rowData[0:15])
            for i in rowData:
                rowScaled.append((i-minMonth)/(maxMonth-minMonth))

            
            if (rowScaled[15] < rowScaled[14]):
                rowScaled[15] = 0
            else:
                rowScaled[15] = 1
            print(rowScaled)
            writer.writerow(rowScaled)
    

f.close()
g.close()
