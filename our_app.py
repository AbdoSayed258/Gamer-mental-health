from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle


filename = 'RandomForestClass.pkl'
clf = pickle.load(open(filename, 'rb'))
le=pickle.load(open('encoder.pkl','rb'))
app = Flask(__name__, template_folder='Templates')



@app.route('/')
def home():
	return render_template('ml2.html')

@app.route('/result',methods=['POST','GET']) #post get
def result():
    if request.method == 'POST':
        # if 'sub' in request.form:
        accept=request.form.get('accept2')
        
        gad1 = request.form.get('gad1')
        gad2 = request.form.get('gad2') 
        gad3 = request.form.get('gad3')
        gad4 = request.form.get('gad4')
        gad5 = request.form.get('gad5')
        gad6 = request.form.get('gad6')
        gad7 = request.form.get('gad7')
        gade = request.form.get('gade')
        
        gad_t = int(gad1) + int(gad2) + int(gad3) + int(gad4) + int(gad5) +int(gad6)+int(gad7)
        
        swl1= request.form.get('SWL1')
        swl2= request.form.get('SWL2')
        swl3= request.form.get('SWL3')
        swl4= request.form.get('SWL4')
        swl5= request.form.get('SWL5')
       
        swl_t = int(swl1)+int(swl2)+int(swl3)+int(swl4)+int(swl5)
        
        game = request.form['game']
        
       
        
        platform = request.form.get('pf')
        
        earnings = request.form.get('hb')
        
        whyplay = request.form.get('whyplay')
        
        spin1 = request.form.get('spin1')
        spin2 = request.form.get('spin2')
        spin3 = request.form.get('spin3')
        spin4 = request.form.get('spin4')
        spin5 = request.form.get('spin5')
        spin6 = request.form.get('spin6')
        spin7 = request.form.get('spin7')
        spin8 = request.form.get('spin8')
        spin9 = request.form.get('spin9')
        spin10 = request.form.get('spin10')
        spin11 = request.form.get('spin11')
        spin12 = request.form.get('spin12')
        spin13 = request.form.get('spin13')
        spin14 = request.form.get('spin14')
        spin15 = request.form.get('spin15')
        spin16 = request.form.get('spin16')
        spin17 = request.form.get('spin17')
        
        spin_t =(int(spin1) + int(spin2) + int(spin3) + int(spin4) + int(spin5) + int(spin6) +
                 int(spin7)+int(spin8)+int(spin9)+int(spin10)+int(spin11)+int(spin12)+int(spin13)+int(spin14)+
                 int(spin15)+int(spin16)+int(spin17))
        
        narcissism = request.form.get('nar')
        age = request.form['age']        
        age=int(age)
        
        
        
        data = [gad_t , swl_t , spin_t , narcissism , age ]
        # transformed = le.transform([platform , earnings , whyplay])
        predictions = clf.predict([data])
        status = None
        
        # # Observations : 

# #         GAD, SWL  , SPIN
# # Label 0: 6 , 17.5 ,  22  -> Have Social phobia but happy with it 
# # Label 1: 3 , 24   ,  9   -> Have Normal life
# # Label 2: 9 , 16   ,  42  -> Have very high social phobia which affects your satisfication with life
# # '''
        
        
        if predictions ==0:
            status = "You are classified as People who are having Social Phobia but you are still happy with your life"
        elif predictions ==1:
            status = "You are statistics are amazing you aren't classified with any disorder"
        else:
            status ="You Have very high social phobia which affects your satisfication with life , Try to have a Life"
    return render_template('index.html', label = status)
# ,prediction = my_prediction 
# acceptidk = accept, gadtotall=gad_t, gadee=gade, swltotal= swl_t ,gamme=game,plat=platform,err = earnings ,yplay= whyplay,narrr=narcissism,spintotall=spin_t,agge=age


if __name__ == '__main__':
	app.run(debug=True)
