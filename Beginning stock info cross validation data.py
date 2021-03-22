import yfinance as yf
import csv


g = open('sp500tickers.csv')
reader = csv.reader(g)
f = open('Stock cross validation set.csv', 'a')
writer = csv.writer(f)

for row in reader:
    
    
    ticker = row[0]
    print("Writing data for " + ticker)
    yfTicker = yf.Ticker(ticker)
    
    #Get stock info for month of Jan 2015; first 15 entries
    #y label is 1 if last day of feb. price is above last day of jan. price
        #and false otherwise
    hist = yfTicker.history(start='2021-01-15',end='2021-03-13')
    if (hist.shape[0] >= 16):
        rowData = []
        for i in range(0, 15):
            closePrice = hist.iloc[i,3]
            rowData.append(closePrice)

        lastClosePrice = hist.iloc[hist.shape[0]-1,3]
        rowData.append(lastClosePrice)

        minMonth = min(rowData[0:15])
        maxMonth = max(rowData[0:15])
        rowScaled = [ticker]
        for i in range(0, 16):
            rowScaled.append((rowData[i]-minMonth)/(maxMonth-minMonth))
        print(rowScaled)
        
        if (rowScaled[16] < rowScaled[15]):
            rowScaled[16] = 0
        else:
            rowScaled[16] = 1
        print(rowScaled)
        writer.writerow(rowScaled)
    

f.close()
g.close()
