import tabula
import dateutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET 
from flask import Flask, request, render_template,Response,jsonify
import pickle

app = Flask(__name__)

#Loading a model:
model = pickle.load(open('model_cv_iso.pkl', 'rb'))

ALLOWED_EXTENSIONS = set(['xls', 'pdf'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#This function read cibil xm file and return data:
def CIBIL(file):
    # create element tree object 
    tree = ET.parse(file) 
    # get root element 
    root = tree.getroot()
    context = root.find('ContextData')
    cibil = context.getchildren()[0].find('Applicants').find('Applicant').find('DsCibilBureau')
    credit_report = cibil.find('Response').find('CibilBureauResponse').find('BureauResponseXml').find('CreditReport')
    name_segment = credit_report.findall('NameSegment')[0]
    id_segment = credit_report.findall('IDSegment')[0]
    tele_segment = credit_report.findall('TelephoneSegment')[0]
    email_segment = credit_report.findall('EmailContactSegment')[0]
    addresses = credit_report.findall('Address')[0]
    score_segment = credit_report.find('ScoreSegment')
    accounts = credit_report.findall('Account')[0]
    enquiries = credit_report.findall('Enquiry')
    #NameSegment:
    name1 = name_segment.find('ConsumerName1').text
    name2 = name_segment.find('ConsumerName2').text
    dob = name_segment.find('DateOfBirth').text
    dob = '-'.join([dob[:2],dob[2:4],dob[4:]])
    gender = name_segment.find('Gender').text
    if int(gender) == 1:
        gender = 'Female'
    else:
        gender = 'Male'

    #IDSegment:
    pan_no = id_segment.find('IDNumber').text

    #Telephone Segment:
    phone_no = tele_segment.find('TelephoneNumber').text
    
    #Email Segment:
    email = email_segment.find('EmailID').text
    
    #Score Segement:
    cibilscore = int(score_segment.find('Score').text)
    
    #Address Segment:
    a1 = addresses.find('AddressLine1').text
    a2 = addresses.find('AddressLine2').text
    address = a1+', '+a2
    pin = addresses.find('PinCode').text
    
    #Account Segment:
    details = accounts.find('Account_NonSummary_Segment_Fields')
    
    try:
        ac_no = details.find('AccountNumber').text
    except:
        ac_no = '-'
        
    try:
        open_date = details.find('DateOpenedOrDisbursed').text
    except:
        open_date = '-'
        
    if open_date != '-':
        open_date = '-'.join([open_date[:2],open_date[2:4],open_date[4:]])
        
    try:
        last_date = details.find('DateOfLastPayment').text
    except:
        last_date = '-'
        
    if last_date != '-':
        last_date = '-'.join([last_date[:2],last_date[2:4],last_date[4:]])
        
    try:
        amount = details.find('HighCreditOrSanctionedAmount').text
    except:
        amount = '-'
        
    try:
        balance = details.find('CurrentBalance').text
    except:
        balance = '-'
        
    try:
        overdue = details.find('AmountOverdue').text
    except:
        overdue = '-'
        
    try:
        interest = details.find('RateOfInterest').text
    except:
        interest = '-'
        
    try:
        tenure = details.find('RepaymentTenure').text
    except:
        tenure = '-'
    
    try:
        emi = details.find('EmiAmount').text
    except:
        emi = '-'
        
    try:
        collateral_Value = details.find('ValueOfCollateral').text
    except:
        collateral_Value = '-'
        
    #EnquirySegment:
    total_no_enquiries = len(enquiries)
    
    enquiry = enquiries[0]
    
    try:
        last_enq_date = enquiry.find('DateOfEnquiryFields').text
    except:
        last_enq_date = '-'
    
    if last_enq_date != '-':
        last_enq_date = '-'.join([last_enq_date[:2],last_enq_date[2:4],last_enq_date[4:]])
        
    try:
        last_enq_purpose = enquiry.find('EnquiryPurpose').text
    except:
        last_enq_purpose = '-'
    
    try:
        last_enq_amt = enquiry.find('EnquiryAmount').text
    except:
        last_enq_amt = '-'
        
    whole_data = [name1, 
                name2,
                dob,
                gender,
                pan_no,
                phone_no,
                email,
                cibilscore,
                address,
                pin,
                ac_no,
                open_date,
                last_date,
                amount,
                balance,
                overdue,
                interest,
                tenure,
                emi,
                collateral_Value,
                total_no_enquiries,
                last_enq_date,
                last_enq_purpose,
                last_enq_amt]
    
    return whole_data
    
#This is a function to read the pdf bank statement of HDFC Bank:
def HDFC_PDF(file):
    """
    This function wil parse the HDFC PDF Statement.
    This function will return the average bank balance per month.
    """
    #Read PDF file:
    tables = tabula.read_pdf(file,pages='all')

    #Combining all tables:
    table = []
    for i in range(len(tables)):
        s1 = tables[i].values.tolist()
        table.extend(s1)


    #Removing unwanted columns:
    ex = []
    for i in range(len(tables)):
        if tables[i].shape[1] == 7:
            ex.extend(tables[i].values.tolist())
        elif tables[i].shape[1] == 6:
            table = tables[i].values.tolist()
            for i in table:
                i.append(i[5])
                i[5] = np.nan
                ex.append(i)

        elif tables[i].shape[1] == 8:
            table = tables[i].values.tolist()
            for i in table:
                del i[2]
                ex.append(i)

    #Creating dataframe:
    df = pd.DataFrame(ex,columns=['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance'])

    #Removing rows which having date is null:
    df = df[~df['Date'].isnull()]

    #Parsing Closing Price
    df["Closing Balance"] = df["Closing Balance"].astype(str)

    #Converting dataset into List:
    l1 = df.values.tolist()

    #Handiling Closing Balance column:
    final = []
    for i in l1:
        splits = (i[-1].split())
        if (len(splits)>1):
            i[-2] = splits[0]
            i[-1] = splits[1]
            final.append(i)
        else:
            final.append(i)

    #Creating dataframe:
    final = pd.DataFrame(final,columns=['Date', 'Narration', 'Chq/Ref.No', 'Value Dt', 'Withdrawal Amt','Deposit Amt', 'Closing Balance'])

    #Parsing the date fields:
    final['Date'] = final['Date'].apply(dateutil.parser.parse, dayfirst=True)
    final['Value Dt'] = final['Value Dt'].apply(dateutil.parser.parse, dayfirst=True)

    #Paring prices:
    final['Closing Balance'] = final['Closing Balance'].astype(str)
    col = ['Closing Balance']
    for i in col:
        val = []
        for j in final[i]:
            val.append(''.join(j.split(',')))
        final[i] = val

    #TypeCasting Closing Balance:
    col = ['Closing Balance']
    for i in col:
        final[i] = pd.to_numeric(final[i],errors='coerce')

    #Group by operation to close price:
    group = final.groupby(pd.Grouper(key='Date',freq='1M'))

    #Filtering close balance per month:
    balance_month = []
    for i in group:
        a = i[1]
        balance_month.append(a['Closing Balance'].iloc[-1])

    #Closing Balance Per Month:
    return np.average(balance_month)

@app.route('/file-upload', methods=['POST','GET'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    #Uploading file:
    file = request.files['file']
    
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    
    if file and allowed_file(file.filename):
        resp = jsonify({'message' : 'File successfully uploaded'})
        resp.status_code = 201
        tables = tabula.read_pdf(file,pages='all')
        table = []
        for i in range(len(tables)):
            s1 = tables[i].values.tolist()
            table.extend(s1)
        ex = []
        for i in range(len(tables)):
            if tables[i].shape[1] == 7:
                ex.extend(tables[i].values.tolist())
             
            elif tables[i].shape[1] == 6:
                table = tables[i].values.tolist()
                for i in table:
                    i.append(i[5])
                    i[5] = np.nan
                    ex.append(i)
                    
            elif tables[i].shape[1] == 8:
                table = tables[i].values.tolist()
                for i in table:
                    del i[2]
                    ex.append(i)
        df = pd.DataFrame(ex,columns=['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance'])
        return Response(
            df.to_csv(index=False),
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename=result.csv"})
    else:
        resp = jsonify({'message' : 'Allowed file types are xls, pdf'})
        resp.status_code = 400
        return resp
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/back',methods=['POST','GET'])
def back():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    res = []
    
    #Uploading file:
    cibil_file = request.files['cibil']   
    destination = cibil_file
    cibil_data = CIBIL(destination)
    
    status = request.form["martial_status"]
    married = {2750:"Married",2751:"Un Married"}
    predict_request.append(status)
    res.append(married.get(int(status)))
    
    dep = request.form["dependants"]
    predict_request.append(dep)
    res.append(dep)
    
    resi = request.form["residence"]
    residence = {2755:"Own",2756:"Rent"}
    predict_request.append(resi)
    res.append(residence.get(int(resi)))
    
    year = request.form["staying_year"]
    predict_request.append(year)
    res.append(year)
    
    age = request.form["age"]
    predict_request.append(age)
    res.append(age)
    
    indus = request.form["industrytype"]
    ind_cat = {1782:"Salaried",1783:"Self Employeed",603:"Agriculture",
     604:"Passenger Transportation",605:"Construction",875:"Infrastructure",
     876:"Cement",877:"Oil and Gas",878:"Government Contract",879:"Others",658:"Mine"}
    predict_request.append(indus)
    res.append(ind_cat.get(int(indus)))
    
    profile = request.form["profile"]
    pro_cat = {2692:"Captive Class",2693:"Retail Class",2694:"Strategy Class"}
    predict_request.append(profile)
    res.append(pro_cat.get(int(profile)))
    
    segment = request.form["segment"]
    seg_cat = {2695:"First Time Buyer",
            2696:"First Time Buyer Plus",
            2697:"Medium Fleet Operators",
            2698:"Small Fleet Operators"}
    predict_request.append(segment)
    res.append(seg_cat.get(int(segment)))
    
    market = request.form["market_load"]
    market_cat = {566:"Market Load",568:"Own Contract",569:"Attached To Fleet Operator"}
    predict_request.append(market)
    res.append(market_cat.get(int(market)))
    
    years = request.form["tot_years"]
    predict_request.append(years)
    res.append(years)
    
    asset = request.form["asset"]
    predict_request.append(asset)
    res.append(asset)
    
    
    cat = request.form["productcat"]
    prod_cat = {1784:"Loan Against Property",
            926:"Car",
            912:"Multi Utility Vehicle",
            945:"Vikram",
            1402:"Tractor",
            1373:"Used Vehicles",
            1672:"Tipper",
            1664:"Farm Equipment",
            1541:"Two Wheeler",
            634:"Intermediate Commercial Vehicle",
            527:"Heavy Commercial Vehicle",
            528:"Construction Eqquipments",
            529:"Three Wheelers",
            530:"Light Commercial Vehicle",
            531:"Small Commercial Vehicle",
            738:"Medium Commercial Vehicle",
            783:"Busses"}
    predict_request.append(cat)
    res.append(prod_cat.get(int(cat)))
    
    brand = request.form["brand"]
    brand_type = {
           564:"Ashok Leyland",
           565:"Tata Motors",
           740:"Caterpillar",
           739:"Ace",
           741:"Escorts Ltd",
           743:"JCB",
           744:"L&T-Komatsu",
           745:"L&T-Case",
           723:"Eicher Motors",
           815:"JCBL",
           904:"Hyundai Construction Equipments Ltd",
           1377:"ELGI",
           1433:"Ajax Fiori Engineering Ltd",
           1476:"Hercles",
           1416:"Mahindra Navistar Automotives Ltd",
           1501:"Mahindra Construction Equpiment",
           1620:"Dossan",
           1623:"Liugong",
           1638:"Kobelco Construction Equipment Ltd",
           1639:"Sany",
           1659:"Tata Hitachi",
           1681:"Case New Holland",
           1693:"Volvo",
           1380:"Kirloskar",
           1710:"Action Construction Equipment Ltd",
           1758:"Wirtgen India Ltd",
           1760:"Komatsu Ltd",
           1768:"L&T Case Equipment Ltd",
           1778:"Tata Hitachi Construction Machinery Ltd",
           1781:"Larsen & Tourbo Ltd",
           1816:"Atlas Copco Ltd",
           1839:"Scania Commercial Vehicles",
           1848:"KYB Conmat",
           1861:"Man Trucks Ltd",
           1864:"Sany Heavy Industry Ltd",
           1868:"Jackson Ltd",
           1985:"Bharat Benz",
           2143:"Bull Machines Ltd",
           2149:"Terex India Ltd",
           2399:"Rock Master",
           2420:"Vermeer",
           2424:"Jiangsu Dilong Heavy Machinery Ltd",
           2818:"Dynapac Road Constrcution Ltd",
           2886:"Kesar Road Equipments"}
    predict_request.append(brand)
    res.append(brand_type.get(int(brand)))
    
    tenure = request.form["tenure"]
    predict_request.append(tenure)
    res.append(tenure)
    
    instal = request.form["instalcount"]
    predict_request.append(instal)
    res.append(instal)
    
    chasasset = request.form["chasasset"]
    predict_request.append(chasasset)
    res.append(chasasset)
    
    chasinitial = request.form["chasinitial"]
    predict_request.append(chasinitial)
    res.append(chasinitial)
    
    chasfin = int(chasasset) - int(chasinitial)
    predict_request.append(chasfin)
    res.append(chasfin)
    
    fininter = request.form["finaninterest"]
    predict_request.append(fininter)
    res.append(fininter)
    
    interestamount = (int(chasfin)*(int(tenure)/12)*(float(fininter)))/100
    emi = (int(chasfin)+int(interestamount))/int(tenure)
    predict_request.append(int(emi))
    res.append(int(emi))
    
    gross_loan = cibil_data[13]
    predict_request.append(gross_loan)
    res.append(gross_loan)
    
    income = request.form["totincome"]
    predict_request.append(income)
    res.append(income)
    
    expense = request.form["totexpense"]
    predict_request.append(expense)
    res.append(expense)
    
    surplus = int(income) - int(expense)
    predict_request.append(surplus)
    res.append(surplus)
    
    s1 = request.form["vehicle"]
    s1_cat = {"1":"New Vehicle","2":"Used Vehicle"}
    res.append(s1_cat.get(s1))
    if (int(s1) == 1):
        predict_request.append(0)
        res.append(0)
    else:
        veh_age = request.form["vehicle_age"]
        predict_request.append(veh_age)
        res.append(veh_age)
    
    #Uploading file:
    file = request.files['file']
    filename = file.filename
    extn = filename.split('.')[-1]   
    destination = file
  
    #Checking for extension of file: 
    if (extn.casefold() == 'pdf'):
        #Returned a result from a function calling:
        clobal =  HDFC_PDF(destination)
    
    if (extn.casefold() == 'xls'):
        #Loading dataset:
        df = pd.read_excel(destination)
        
        #Fetching transactions only:
        row_no = 0
        for i in df.iloc[:,0]:
            if i == 'Date':
                df = df.iloc[row_no:]
                break
            row_no = row_no+1
        
        #Set a features name:
        df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance']
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        #Dropping first two records:
        df.drop([0,1],axis=0,inplace=True)
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        row_no = 0
        for i in df['Date']:
            if len(str(i)) != 8:
                df = df.iloc[:row_no]
                break
            row_no = row_no + 1
            
        # Parsing date:
        df['Date'] = df['Date'].apply(dateutil.parser.parse, dayfirst=True)
        table = df
        
        #Group by operation to find opening and close price:
        group = table.groupby(pd.Grouper(key='Date',freq='1M'))
        
        #Filtering open and close balance per month:
        balance_month = []
        for i in group:
            a = i[1]
            balance_month.append(a['Closing Balance'].iloc[-1])
        
        clobal = (np.average(balance_month))
   
    predict_request.append("{:.2f}".format(clobal))
    res.append("{:.2f}".format(clobal))
    
    score = request.form["score"]
    predict_request.append(score)
    res.append(score)
    
    another_score = cibil_data[7]
    predict_request.append(another_score)
    res.append(another_score)
    
    ###############################
    loan = ((int(chasfin)*100)/int(chasasset))
    if (loan<75):
        predict_request.append(100)
        res.append(100)
    elif ((loan>=75) and (loan <=80)):
        predict_request.append(75)
        res.append(75)
    elif (loan>80):
        predict_request.append(0)
        res.append(0)
        
    
    if int(asset) > (2*chasfin):
        predict_request.append(120)
        res.append(120)
    elif int(asset) == (2*chasfin):
        predict_request.append(75)
        res.append(75)    
    elif int(asset) == chasfin:
        predict_request.append(40)
        res.append(40)    
    else:
        predict_request.append(0)
        res.append(0)
        
    od = [1 if int(cibil_data[15])>0 else 0][0]
    od_cat = {0:"No",1:"Yes"}
    res.append(od_cat.get(int(od)))
    if int(od) == 0:
        predict_request.append(100)
        res.append(100)
    elif int(od) == 1:
        predict_request.append(0)
        res.append(0)
        
    bank_p = request.form["bank_period"]
    bank_p_cat = {1:"More than 3 years",
                  2:"Between 1 to 3 years",
                  3:"More than 6 months",
                  4:"Less than 6 months"}
    res.append(bank_p_cat.get(int(bank_p)))
    if int(bank_p) == 1:
        predict_request.append(100)
        res.append(100)
    elif int(bank_p) == 2:
        predict_request.append(50)
        res.append(50)
    elif int(bank_p) == 3:
        predict_request.append(0)
        res.append(0)
    elif int(bank_p) == 4:
        predict_request.append(-50)
        res.append(-50)
        
    if int(market) == 569 and int(segment) == 2697:
        predict_request.append(60)
        res.append(60)
    
    elif int(market) == 569 and int(segment) == 2698:
        predict_request.append(25)
        res.append(25)
            
    elif int(market) == 566:
        predict_request.append(0)
        res.append(0)
    
    elif int(market) == 568:
        predict_request.append(80)
        res.append(80)
    else:
        predict_request.append(0)
        res.append(0)
       
    ##############################
    
    gender_dict = {'M':[0,1],'F':[1,0]}
    cate = request.form["gender"]
    if cate == 'M':
        res.append('Male')
    else:
        res.append('Female')
    res.append(request.form["pan"])
    
      
    if int(segment) == 2695:
        res.append(0)
    else:
        #Getting fleet size:
        fleet = request.form["fleet"] 
        res.append(fleet)
        
    predict_request.extend(gender_dict.get(cate))
    predict_request = list(map(float,predict_request))
    predict_request = np.array(predict_request)
    prediction = model.predict_proba([predict_request])[0][-1]
    output = int((1 - prediction)*100)
    if output < 60:
        condition = 'Risky'
    if output >= 60 and output <= 70:
        condition = 'Barely Acceptable'
    if output >= 71 and output <= 80:
        condition = 'Medium'
    if output >= 81 and output <=90:
        condition = 'Good'
    if output > 90:
        condition = 'Superior' 
    return render_template('resultpage.html', prediction_text=output,data=res,status=condition,cibil = cibil_data[7],info = cibil_data)

if __name__ == "__main__":
    app.run(debug=True)
