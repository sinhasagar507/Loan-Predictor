"""
@author:Sagar Sinha

"""
from flask import Flask, render_template, request
from flask_cors import cross_origin
import joblib 
import numpy as np


#Initialising the app and loading the models
app = Flask(__name__) 
xgbr_model = joblib.load("./saved models/xgbr.pkl")
lgbm_model = joblib.load("./saved models/lgbm.pkl")
rmr_model = joblib.load("./saved models/rmr.pkl")
print("Models Loaded")
    
#Use flask_cors or secret key
@app.route("/", methods=['GET']) #This signifies the default route
@cross_origin()
def home():
    return render_template("index.html", title="Home Page")

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
         emi = float(request.form['EMI'])     # Monthly EMI
         income = float(request.form['Income'])   # Income
         ltv = float(request.form['LTV']) # Loan to Value Ratio
         assetCost = float(request.form['Asset-Cost'])  # Asset Cost
         age = float(request.form['Age'])  # Age
         meanBal = float(request.form['Balance-Mean'])   # Mean Balance
         currBal = float(request.form['Current-Bal'])  # Maximum Current Balance 
         disbursedAmt = float(request.form['Disbursed-Amt'])   # Disbursed Amount
         amtFin = float(request.form['Amt-Fin'])  # Finance-Amount
         minCBS = float(request.form['Min-CBS']) # Minimum Current Balance Sum
         delays = float(request.form['Delays'])   # Delays in request 
         paymentPD = float(request.form['Payment-PP'])  # Count of Payments Past Due Dates
         meanSCB = float(request.form['Mean-SCB'])   # Mean of Successive Current Balance Differences
         sumDelays = float(request.form['Delays-Sum'])  # Sum of number of delays in payment
         minCB = float(request.form['Min-CB'])   # Minimum Current Balance
         meanMC = float(request.form['MC'])   # Mean of month count with no history of prior payment
         maxCB = float(request.form['Max-CB'])   # Maximum Current Balance
         hout = float(request.form['HOUT'])  # Mean of Maximum Current Balances for no recorded Payment History
         mamiCB = float(request.form['MaMiCB']) # Maximum of Minimum of Current Balances
         cb = float(request.form['CB'])   # Current Balance
         tenure = float(request.form['Tenure'])  # Tenure
         meanBD = float(request.form['Mean-BD']) # Mean of Sum of Successive Current Balance Differences
         mchprp = float(request.form['MCHPRP'])  # Sum of month count with no history of prior payment
         mcppdd = float(request.form['MCPPDD'])  # Maximum Count of Payment Past Due Date
         mcnpaym = float(request.form['MCNPAYM'])  # Maximum Count of Number of delays in Payment
         hhopaym = float(request.form['HHOPAYM'])  # 0 or 1 as per the history of outstanding payment
         mmaapa= float(request.form['MMaAPa']) # Mean of Maximum Amount Paid
         mosoap = float(request.form['MOSOAP'])  # Mean of Sum of Amount Paid 
         mamaap= float(request.form['MaMaAP'])  # Maximum of Maximum Amount Paid
         hhopay = float(request.form['HHOPAY'])  # 0 or 1 as per the history of outstanding payments
         masap = float(request.form['MASAP']) # Maximum of Sum of Amount Paid
         mcndp = float(request.form['MCNDP'])  # Minimum Count of No. of Delays in Payment
         masdcb = float(request.form['MASDCB'])  # Maximum of Average Successive Difference in Current Balance
         minicb = float(request.form['MiniCB']) # Minimum Current Balance 
         mssdcb = float(request.form['MSSDCB']) # Minimum Current Balance
         hhopa = float(request.form['HHOPA'])   # 0 or 1 as per the history of outstanding payment
         mmcpppd = float(request.form['MMCPPPD']) # Minimum Count of Payments Past Due Date 
         memcb = float(request.form['MeMCB'])     # Mean of Minimum Current Balance
         mmcb = float(request.form['MMCB'])  # Minimum of Minimum Current Balance
         o1h0 = float(request.form['01HO'])  # 0 or 1 as per the history of outstanding payments
         smcb = float(request.form['SMCB'])   # Sum of Minimum Current Difference in Balance 
         mimap = float(request.form['MiMAP']) # Minimum of all Amounts Paid 
         smap = float(request.form['SMAP'])          # Sum of Minimum Amount Paid 
         mmap = float(request.form['MMAP']) # Maximum of Minimum Amount Paid   
         miss = float(0) #Take care of the missing value

         input_lst = [emi, income, ltv, assetCost, age, meanBal, currBal, disbursedAmt, 
                     amtFin, minCBS, delays, paymentPD, meanSCB, sumDelays, minCB, meanMC, maxCB, 
                     hout, mamiCB, cb, tenure, meanBD, mchprp, mcppdd, mcnpaym, hhopaym,
                     mmaapa, mosoap, mamaap, hhopay, masap, mcndp, masdcb, minicb, mssdcb, hhopa, 
                     mmcpppd, memcb, mmcb, o1h0, smcb, mimap, smap, mmap, miss]
         
         input_arr = np.array(input_lst).reshape(1, -1)
        
         pred = lgbm_model.predict(input_arr)[0]
         output = int(pred)
         print(output)
         
         if output == 0:
             return render_template("month-1.html")
         
         elif output == 1:
             return render_template("month-2.html")
         
         elif output == 2:
             return render_template("month-3.html")
         
         elif output == 3:
             return render_template("month-4.html")
         
         elif output == 4:
             return render_template("month-5.html")
         
         elif output == 5:
             return render_template("month-6.html")
         
         elif output == 6:
             return render_template("month-7.html")
    
    return render_template("predictor.html", title="Prediction Page")

# Run the app from main method
if __name__ == '__main__':
   app.run(debug=True)
