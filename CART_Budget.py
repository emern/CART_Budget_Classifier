#Budget Processing Program By: Emery Nagy
#inputs: Credit Card data and Banking data in CSV format
#Outputs: Pie Chart Showing Spending Practices, income and spending breakdown sheet

import pandas as pd
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np
import datetime
import nltk
import re
import os
import tkinter as tk
from tkinter import filedialog
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from string import digits
from tkcalendar import DateEntry, Calendar
from tkinter import ttk
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta

def updateDict(dic):
    f = open('dict.txt','w')
    f.write(str(dic))
    f.close()

def load_Dict():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
    return eval(data)

dictionary = load_Dict() #load dictionary used by parser and ML Tree

def init_set(): #initialllize vectorizer Decision Tree
    global clf,cv
    clf = DecisionTreeClassifier(min_samples_split = 3, max_depth = 75, class_weight = "balanced")
    cv = TfidfVectorizer(stop_words = 'english')

def load_training_data(filepath): #Load training data set
    global train
    train = pd.read_excel (filepath , names = ['Date', 'Trans_Type', 'Amount', 'Label'])
    train = train.drop_duplicates(inplace = False) #drop duplicate entries just in case
    train.sort_values('Date')#sort values by date (not necessary, mostly for debugging)
    train['Label'] = train['Label'].map(dictionary) #labels to equivelent values

    train['Date'] = train['Date'].dt.strftime("%a %d %b %Y")
    train['Date'] = train['Date'].str.strip()
    train['Trans_Type'] = train['Trans_Type'].str.strip()
    train['Amount'] = abs(train['Amount'])
    train = train.reset_index(drop = True)
 #send all information to string and strip sting to be vectorized (new version is not using the amount or date)


def preprocess_CC(CC_data, bank):
    CC_data = CC.drop_duplicates(inplace = False)
    CC_data.sort_values('Date')
    CC_data = CC_data.drop(CC_data.loc[(CC_data['Trans_Type'].str.contains("PAYMENT - THANK YOU"))].index) #remove my credit card payments
    try:
        returns = CC_data.loc[(CC_data['Trans_Type'].str.contains("CREDIT VOUCHER/RETURN"))].index #find and append all credit card returns to income csv
        print(returns)
        r_a = returns["Amount"]
        r_d = returns["Date"]
        r_t = returns["Trans_Type"]
        r = pd.concat([r_d, r_t, r_a]).reset_index(drop = True)
        r.to_csv(r'your income csv here.csv', mode='a', header=False, index=False)
    except:
        pass
    CC_data = CC_data.drop(CC_data.loc[(CC_data['Trans_Type'].str.contains("CREDIT VOUCHER/RETURN"))].index)
    CC_data['Label'] = CC_data['Label'].map(dictionary) #map labels
    CC_data['Trans_Type'] = CC_data['Trans_Type'].str.strip()
    CC_data['Amount'] = abs(CC_data['Amount'])
    CC_data = CC_data.reset_index(drop = True)
    CC_data = pd.concat([CC_data, bank]) #put banking and credit card purchases together
    CC_data = CC_data.reset_index(drop = True)
    global Purchases
    Purchases = CC_data
    Purchases = Purchases.drop('Label', axis = 1, inplace = False) #global purchases exists to be used with the processed data at the end

    CC_data['Date'] = CC_data['Date'].dt.strftime("%a %d %b %Y")
    CC_data['Date'] = CC_data['Date'].str.strip()
    CC_data = CC_data.drop('Amount', axis = 1, inplace = False) #pass data out to vectorizer
    return CC_data

def process_bank(df): #process bank statments
    #global purchase_df, salary_df, interest_df, deposit_df
    df = df.drop_duplicates(inplace = False)

    deposits_1 = df.loc[(df['Trans_Type'].str.contains("ATM Deposit"))]
    deposits_2 =  df.loc[(df['Trans_Type'].str.strip() == "Deposit")]
    salary_df = df.loc[(df['Trans_Type'].str.contains("Payroll"))]
    interest_df = df.loc[(df['Trans_Type'].str.contains("Interest"))]
    withdraw = df.loc[(df['Trans_Type'].str.contains("Withdraw"))]

    #filter income types and withdrawels

    deposit_df = pd.concat([deposits_1, deposits_2])
    deposit_df = deposit_df.append([salary_df, interest_df]) #add deposits, salary and interest together
    deposit_df = deposit_df.drop(["Trans #", "Remaining", "Label"], axis = 1)
    splitter_row = pd.DataFrame({'Date': 'X'}, index = [0]) #add splitter row to make going through raw data easier
    deposit_df = pd.concat([splitter_row, deposit_df]).reset_index(drop = True)

    deposit_df.to_csv(r'file where you want income to go.csv', mode='a', header=False, index=False)
    #send to income spreadsheet

    purchase_df = df.loc[(df['Trans_Type'].str.contains("Purchase"))]
    purchase_df = pd.concat([purchase_df, withdraw])
    purchase_df = pd.concat([purchase_df['Date'], purchase_df['Trans_Type'], purchase_df['Amount'], purchase_df['Label']], axis = 1)
    purchase_df['Label'] = purchase_df['Label'].map(dictionary)
    purchase_df['Trans_Type'] = purchase_df['Trans_Type'].str.strip()
    purchase_df['Trans_Type'] = purchase_df['Trans_Type'].str.replace('Purchase:',"")
    purchase_df = purchase_df.reset_index(drop = True)
    #compose all purchase data made on my bank account
    return purchase_df


def get_key():
    global keywords, keyDates
    keywords = open("keyphrases.txt", "r")
    keywords = keywords.read().split()
    keyDates = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    #old code used for fitting vectorizer to manual entries (using CountVectorizer)



def vectorize(data, status):
    #vectorizes text entry data. Fields Z and A where reserved for amount and date but these created lots of noise and have since been removed

    X = data["Trans_Type"].str.replace('\d+', '')
    #z = data['Date'].str.replace('\d+', '')

    if status == 0: #if training mode use transform function
        x_traincv = cv.fit_transform(X)
    #    z_traincv = cv2.fit_transform(z)
    else:
        x_traincv = cv.transform(X) #if testing just use regular fit
    #    z_traincv = cv2.transform(z)
    #pull names for purposes of tree ID
    x_names = cv.get_feature_names()
    #z_names = cv2.get_feature_names()
    #a_names = ["Amount"]
    #vector_names = x_names + z_names + a_names
    vector_names = x_names #+ a_names
    #Put into Pandas dfs
    x_vectors = pd.DataFrame(x_traincv.toarray(), columns= x_names)
    #z_vectors = pd.DataFrame(z_traincv.toarray(), columns= z_names)
    #x_train = pd.concat([x_vectors, z_vectors, data['Amount']], axis = 1)
    #x_train = pd.concat([x_vectors, data['Amount']], axis = 1)
    x_train = x_vectors

    return x_train, vector_names #return vectors and associated names

def train_tree_CC(TTS_amount, x_train, CC_data):

    y = CC_data['Label']#train CART
    x_train = x_train.dropna(0)
    clf.fit(x_train, y)

    if TTS_amount > 0: #testing algorithm, change fields to see how to tree performs to random Train Test Split scenarios and graph them
        max_samples = 300 #number of iterations to test
        max_depths = np.linspace(1, 200, max_samples+1, endpoint=True)
        min_samples_splits = np.linspace(0.05, 1.0, max_samples, endpoint=True)
        min_samples_leafs = np.linspace(0.1, 0.5, max_samples, endpoint=True)
        max_features = list(range(1,x_train.shape[1]))
        train_results = []
        rand = 1
        for min_samples in max_features:
            random = max_depths[rand]
            rand = rand+1
            #generate random seed every time for the random state of the dataset
            X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size=TTS_amount, random_state=int(random))
            dt = DecisionTreeClassifier(max_features = min_samples, class_weight = "balanced") #can use this to change any field to be interated through i.e max_depths, min_samples_split
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            train_results.append(metrics.accuracy_score(y_test, y_pred))
            print(metrics.accuracy_score(y_test, y_pred))

        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(max_features, train_results, 'b', label='accuracy')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.title('max features of tree over iterations')
        plt.show()


def generate_tree_graphic_CC(names, dict): #generate tree graphic to show descision tree logic
    classes = list(dict)
    dot_data = tree.export_graphviz(clf , feature_names = names, class_names = classes)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('Tree.png')

def predict_CC(CC_data):
    CC_data = CC_data.to_numpy() #send data to numpy array

    y_pred = clf.predict(CC_data) #predict
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['Prediction']
    reverse = {v: k for k, v in dictionary.items()}
    y_pred = y_pred['Prediction'].map(reverse)

    test_results = pd.concat([Purchases, y_pred], axis = 1)
    splitter_row = pd.DataFrame({'Date': 'X'}, index = [0])
    test_results = pd.concat([splitter_row, test_results]).reset_index(drop = True)

    test_results.to_csv(r'file you want purchase data to go.csv',  mode='a', header=False, index=False)
    #send data to master purchases spreadsheet
    return test_results

def GUI(): #GUI used to load Credit Card and Banking data, also specify date to generate report for
    root= tk.Tk()
    canvas1 = tk.Canvas(root, width = 200, height = 200, bg = 'lightsteelblue2', relief = 'raised')
    canvas1.pack()

    def getCC ():#get CC data
        global CC
        import_file_path = filedialog.askopenfilename()
        CC = pd.read_csv (import_file_path, names = ['Date', 'Trans_Type', 'Amount', 'Label'])
        CC['Date'] = pd.to_datetime(CC['Date'], errors='coerce')
        CC = CC.drop(CC.loc[(CC['Trans_Type'].str.contains("PAYMENT - THANK YOU"))].index)


    browseButton_CSV = tk.Button(text="      Import Credit Card Data     ", command=getCC, bg='green', fg='white', font=('helvetica', 12, 'bold'))
    canvas1.create_window(100, 100, window=browseButton_CSV)

    canvas2 = tk.Canvas(root, width = 200, height = 200, bg = 'lightsteelblue2', relief = 'raised')
    canvas2.pack()

    def getBank (): #get banking data
        global Bank
        import_file_path = filedialog.askopenfilename()
        Bank = pd.read_csv (import_file_path, names = ['Trans #', 'Date', 'Trans_Type', 'Amount', 'Deposit', 'Remaining', 'Label'])
        Bank['Date'] = pd.to_datetime(Bank['Date'], errors='coerce')


    browseButton_Bank = tk.Button(text="      Import Banking Data     ", command=getBank, bg='green', fg='white', font=('helvetica', 12, 'bold'))
    canvas2.create_window(100, 100, window=browseButton_Bank)

    def calendar_view(): #get selected date
        def select():
            global entered_date
            entered_date = cal.selection_get()

        top = tk.Toplevel(root)

        cal = Calendar(top, font="Arial 14", selectmode='day', cursor="hand1", year=2020, month=5, day=5)
        cal.pack(fill="both", expand=True)
        ttk.Button(top, text="ok", command=select).pack()

    ttk.Button(root, text='Calendar', command=calendar_view).pack(padx=10, pady=10)

    root.mainloop()

def get_time(delay): #generate string time and calculate any specified delays
    global obj_Date, curr_Date, obj_prev, prev_month
    obj_Date = entered_date - timedelta(days = delay)
    curr_Date = obj_Date.strftime('%Y %h')

    obj_prev = entered_date - timedelta(days = delay+30)
    prev_month = obj_prev.strftime('%Y %h')

def historical(data): #generate historical report of incomes

    data['Date'] = pd.to_datetime(data['Date'])
    data  = data.drop(data.loc[(data['Date'] < pd.Timestamp(obj_Date - relativedelta(years =+1)))].index) #drop anything over 1 year old
    g = data.groupby(pd.Grouper(key='Date', freq='M')) #group purchases by date-> month
    group_dates = list(g.groups.keys()) #list ID keys
    result = pd.DataFrame()
    length = len(group_dates)

    for x in range(length): #for every group key
        active = g.get_group(group_dates[x]) #get group based on group #
        active = active.set_index('Label')
        sum_Data = active.groupby('Label')['Amount'].sum() #sum values per month
        col = group_dates[x].strftime("%Y %b") #generate column name based on group month
        sum_Data = sum_Data.reset_index() #reset index
        sum_Data.columns = ['Label', col]
        sum_Data = sum_Data.set_index('Label') #use label (i.e gas, food) for row indexer
        result = pd.concat([result , sum_Data],axis = 1) #add group to results
        result = result.fillna(0) #fill any NaN

    result = result.reset_index()
    index = result['index']
    recent = pd.concat([index, result[curr_Date]], axis = 1) #pull current month
    previous = pd.concat([index, result[prev_month]], axis = 1) #pull previous month

    fig, (ax1, ax2) = plt.subplots(1, 2) #create plot with 2 subplots
    numFields = len(recent)
    exp = []
    for x in range(numFields):
            exp.append(0.02) #add to pie chart explode array for every field
    def my_autopct(pct): #delete blank percents
        return ('%.2f' % pct) if pct != 0 else ''
    recent.plot.pie(explode = exp, y=curr_Date, autopct=my_autopct, startangle = 90, pctdistance=1.35, radius = 0.8, labeldistance=None, ax = ax1).legend(bbox_to_anchor=(1, 1.25), labels = recent['index'])
    previous.plot.pie(explode = exp, y=prev_month, autopct=my_autopct, startangle = 90, pctdistance=1.35, radius = 0.8, labeldistance=None, ax = ax2).legend(bbox_to_anchor=(1, 1.25), labels = previous['index'])
    #generate pie charts
    ax1.set_title(str(round(result[curr_Date].sum())))
    ax2.set_title(str(round(result[prev_month].sum()))) #set title as the total

    plt.tight_layout()
    result = result.set_index('index')
    sums = result.sum() #add monthly totals
    sums = sums.to_frame(name = 'Total')
    sums = sums.transpose()
    results = pd.concat([result, sums])
    results['Sum'] = result.sum(axis =1)
    results['Avg'] = result.mean(axis = 1)
    results['Std_dev'] = result.std(axis =1) #add total, average and standard deviation
    return results, fig


def banking(info):
    info['Date'] = pd.to_datetime(info['Date'])
    info = info.drop(info.loc[(info['Date'] < pd.Timestamp(obj_Date - relativedelta(years =+1)))].index)
    g = info.groupby(pd.Grouper(key='Date', freq='M'))
    group_dates = list(g.groups.keys())
    result = pd.DataFrame()
    length = len(group_dates) #group income same as purchases



    for x in range(length):
        active = g.get_group(group_dates[x])
        active = active.set_index('Trans_Type')
        sum_Data = active.groupby('Trans_Type')['Amount'].sum()
        col = group_dates[x].strftime("%Y %b")
        sum_Data = sum_Data.reset_index()
        sum_Data.columns = ['Trans_Type', col]
        sum_Data = sum_Data.set_index('Trans_Type')
        result = pd.concat([result , sum_Data],axis = 1)
        result = result.fillna(0)

    result = result.reset_index()
    index = result['index']
    recent = pd.concat([index, result.iloc[:,len(result.columns) - 1]], axis = 1)
    previous = pd.concat([index, result.iloc[:,len(result.columns) - 2]], axis = 1)
    result = result.set_index('index')
    sums = result.sum()
    sums = sums.to_frame(name = 'Total')
    sums = sums.transpose()
    results = pd.concat([result, sums])
    results['Sum'] = result.sum(axis =1)
    results['Avg'] = result.mean(axis = 1)
    results['Std_dev'] = result.std(axis =1)

    return results


def build_output(income, CC, picture):
    global path
    path = r"file where you would like your ouptut folders to live\{}".format(curr_Date) #generate new computer folder path with the specified month as name
    os.makedirs(path)

    CC.loc['Income'] = income.loc['Total']
    CC.loc['Net'] = income.loc['Total'] - CC.loc['Total']
    #calculate monthly net

    ending = '\Spending.csv'
    cc_address = "".join((path, ending))
    CC.to_csv(cc_address) #create csv file "Spending.CSV"

    image_address = "".join((path, '\purchase.png'))
    picture.savefig(image_address)   # save the figure to file

    bank_ad = "".join((path, '\income.csv'))
    income.to_csv(bank_ad) #create csv file "Income.CSV"

def GUI_2(): #stops program and gives user the time to check if there are any errors with first stage data
    def close_window():
        root.destroy()
    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()

    button = tk.Button(frame, text="Proceed with Sorting", fg="red", command=close_window)
    button.pack(side=tk.LEFT)
    root.mainloop()


GUI()
get_time(0)
dictionary = load_Dict()
init_set()
load_training_data(r'load training.xlsx')
get_key()

train_tree_CC(0, vectorize(train, 0)[0], train)
generate_tree_graphic_CC(vectorize(train, 0)[1], dictionary)
predict_CC(vectorize(preprocess_CC(CC, process_bank(Bank)), 1)[0])

GUI_2()

#load income and purchases, drop duplicates, and drop X lines
Spending = pd.read_csv (r'file where your purchases live.csv' , names = ['Date', 'Trans_Type', 'Amount', 'Label'])
Spending = Spending.dropna(axis = 0)
Spending = Spending.drop(Spending.loc[(Spending['Date'].str.contains("X"))].index)
Spending['Amount'] = Spending['Amount'].abs()
Spending['Date'] = Spending['Date'].str.replace('12:00:00 AM',"")
Spending = Spending.drop_duplicates(subset = ['Trans_Type', 'Date', 'Amount'], inplace = False, keep = 'first')
bank = pd.read_csv(r'file where your income lives.csv', names = ['Date', 'Trans_Type', 'Withdraw', 'Amount'])
bank = bank.drop_duplicates(inplace = False)
Bank = Bank.dropna(axis = 0)
bank = bank.drop(bank.loc[(bank['Date'].str.contains("X"))].index)



build_output(banking(bank),historical(Spending)[0],historical(Spending)[1])
