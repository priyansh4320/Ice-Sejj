import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import nltk
from nltk import word_tokenize, ne_chunk , pos_tag
#, download_dir='/path/to/nltk_data/'    :: path for percentron
nltk.download('averaged_perceptron_tagger')
nltk.download("maxent_ne_chunker")
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,mean_squared_error,mean_absolute_error

#---------------------------------------------------------------------------------------------------------------
# all functions for app

#pipeline for sentiment analysis
@st.cache_resource
def pipe(text):
    pipe_sentiment = pipeline("sentiment-analysis")
    return pipe_sentiment(text)

# function to print summary
@st.cache_resource
def summer(text, sum_size):
    summarizer = LexRankSummarizer()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary_size = sum_size
    summary_sentences = summarizer(parser.document, summary_size)
    summary = ' '.join(str(sentence) for sentence in summary_sentences)
    return summary

#Named Entity Recognition Function
@st.cache_resource
def ner(text):
    text = word_tokenize(text)

    pos_text = pos_tag(text)

    ner = ne_chunk(pos_text)

    
    named_entities = []
    for chunk in ner:
        if hasattr(chunk, 'label') and chunk.label() != 'O':
            entity_label = chunk.label()
            entity_words = ' '.join(c[0] for c in chunk)
            named_entities.append(f'{entity_label}: {entity_words}')
        
    return named_entities

#parts of speech Tagging function
@st.cache_resource
def postag(text):
    text = word_tokenize(text)

    pos_text = pos_tag(text)
    l1=[]
    for token, tag in pos_text:
        l1.append(f"{token}: {tag}")
    return l1




#--------------------------------------------------------------------------------------------------------
#app
st.set_page_config(layout="wide")
st.title("Ice Age")
st.write("A Streamlit Application That Demonstrates Power of Machine Learing.")

with st.sidebar:
    st.title("IceAge")
    st.header("Content Navigation")
nav_selection = st.sidebar.radio('Navigation',("NLP","Classification","Regression","Description"))


#NLP TASK
if nav_selection == "NLP":
    tab1,tab2,tab3,tab4 = st.tabs(['SENTIMENT ANALYSIS','SUMMERIZATION','NAMED ENTITY RECOGNITION','POS TAGGING'])

    with tab1:
        st.title("Sentiment Analysis")

        T1_data = st.text_area("Enter Text For Analysis")

        result_btn = st.button("Sentiment")

        if(result_btn ):
            sent_result = pipe(str(T1_data))
            st.write(sent_result)

    with tab2:
        st.title("Text Summarization")

        T2_data = st.text_area("Enter Text For Summerization")
        sum_size = st.text_input("enter size of summary in no. of lines")

        result_btn = st.button("Summery")

        if(result_btn ):
            summary = summer(str(T2_data),int(sum_size))
            st.write(summary)

    with tab3:
        st.title("Named Entity Recognition")

        T3_data = st.text_area("Enter Text For NER")

        result_btn = st.button("NER Result")

        if(result_btn ):
            result_ner = ner(str(T3_data))
            st.write(result_ner)

    with tab4:
        st.title("Part - Of - Speech Tagging")

        T4_data = st.text_area("Enter Text For POS tag")

        result_btn = st.button("POS TAGS")

        if(result_btn ):
            st.write(postag(str(T4_data)))
            st.image("https://m-clark.github.io/text-analysis-with-R/img/POS-Tags.png")
    pass


#CLASSIFICATION TASK
if nav_selection=="Classification":
    class_tab1,class_tab2 = st.tabs(['LOGISTIC REGRESSION','NAIVE BAYES CLASSIFIER'])

    with class_tab1:
        df,df_cols = [],[]
        st.title("Logistic Regression")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df_cols = [i for i in df.columns]
            st.dataframe(df)
        col1,col2,col3,col4 = st.columns(4)

        with col1:
            btn = st.button("show cloumn name for DF")
            if btn:
                st.write(df.columns)
            
        with col2:
            btn = st.button("show all null values")
            if btn:
                st.write(df.isnull().sum())
            
        with col3:
            btn = st.button("Describe DF")
            if btn:
                st.write(df.describe())
            
        with col4:
            show_type_of_data = st.button("Show Datatypes")
            if show_type_of_data:
                st.write(df.dtypes)
            
        #preprocessing
        st.header("PREPROCESS , TRAIN  AND PREDICT")
        st.write("works for only numeric features")
        

        class_form = st.form(key = 'classification')
        select_feature = class_form.multiselect("select feature  for Feature Engineering", df_cols)
        target = class_form.selectbox("select Target ",df_cols)
        prediction_data = class_form.text_input("enter prediction data  (seperator = ','  and no gaps)")

        train = class_form.form_submit_button(" Preprocess and train LR ")        
        if train:
            if df is not None:
                df_process = df
            df_process = df_process.dropna()        
            
            if select_feature is not None:
                df_process = df_process[select_feature]
            else:
                df_process = df_process
            
            result_col1,result_col2,result_col3 = st.columns(3)
            with result_col1:
                st.write(df_process)
            with result_col2:
                st.write(f"null values:",df_process.isnull().sum())

            y = df_process[[target]]
            x = df_process.drop([target],axis =1)
            with result_col3:
                st.write("y columns: ",y.columns)
                st.write(f"x columns",x.columns)
            
            trainx , testx, trainy, testy = train_test_split(x,y,test_size=0.2,random_state=42)

            regr = LogisticRegression()
            lr_model = regr.fit(trainx,trainy)

            pred = lr_model.predict(testx)

            prediction_data = prediction_data.split(",")
            prediction_data = [float(i) for i in prediction_data]
            prediction_data_result = lr_model.predict([prediction_data])

            st.title(f" predicted column: {target} : for data :{prediction_data} is {prediction_data_result}")

            
            scores = {
            'acc_score' : round(accuracy_score(pred,testy)*100,2),
            'pre_score' : round(precision_score(testy,pred)*100,2),
             'recall_score': round(recall_score(testy,pred)*100,2)
            }

            
            st.write(scores)
            #show_score = st.button("show accuracy")

            
                
        pass
    pass

    with class_tab2:
        df,df_cols = [],[]
        st.title("Naive Bayes Classifier")
        uploaded_file_naive = st.file_uploader("Upload any CSV file", type=["csv"])
        if uploaded_file_naive is not None:
            df = pd.read_csv(uploaded_file_naive)
            df_cols = [i for i in df.columns]
            st.dataframe(df)
        ncol1,ncol2,ncol3,ncol4 = st.columns(4)

        with ncol1:
            btn = st.button("show cloumn names for DF.")
            if btn:
                st.write(df.columns)
            
        with ncol2:
            btn = st.button("show all null value.")
            if btn:
                st.write(df.isnull().sum())
            
        with ncol3:
            btn = st.button("Describe DF.")
            if btn:
                st.write(df.describe())
            
        with ncol4:
            show_type_of_data = st.button("Show Datatype.")
            if show_type_of_data:
                st.write(df.dtypes)
        
        #preprocessing
        st.header("PREPROCESS , TRAIN  AND PREDICT")
        st.write("works for only numeric features")
        

        nform = st.form(key="naive")
        select_feature = nform.multiselect("select feature  for Feature Engineering", df_cols)
        target = nform.selectbox("select Target ",df_cols)
        prediction_data = nform.text_input("enter prediction data  (seperator = ','  and no gaps)")

        train = nform.form_submit_button(" Preprocess and train NB ")        
        if train:
            if df is not None:
                df_process = df
            df_process = df_process.dropna()        
            
            if select_feature is not None:
                df_process = df_process[select_feature]
            else:
                df_process = df_process
            
            result_col1,result_col2,result_col3 = st.columns(3)
            with result_col1:
                st.write(df_process)
            with result_col2:
                st.write(f"null values:",df_process.isnull().sum())

            y = df_process[[target]]
            x = df_process.drop([target],axis =1)
            with result_col3:
                st.write("y columns: ",y.columns)
                st.write(f"x columns",x.columns)
            
            trainx , testx, trainy, testy = train_test_split(x,y,test_size=0.2,random_state=42)

            classifier = MultinomialNB()
            nb_model = classifier.fit(trainx,trainy)

            pred = nb_model.predict(testx)

            prediction_data = prediction_data.split(",")
            prediction_data = [float(i) for i in prediction_data]
            prediction_data_result = nb_model.predict([prediction_data])

            st.title(f" predicted column {target} : for data : {prediction_data} is {prediction_data_result}")

            
            scores = {
            'acc_score' : round(accuracy_score(pred,testy)*100,2),
            'pre_score' : round(precision_score(testy,pred)*100,2),
             'recall_score': round(recall_score(testy,pred)*100,2)
            }
            st.write(scores)
        pass

if nav_selection=="Regression":
    df,df_cols=[],[]
    regr_tab1 = st.tabs(['LINEAR REGRESSION'])
    st.title("Linear Regression")

    if regr_tab1 :
        uploaded_file_regr = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file_regr is not None:
            df = pd.read_csv(uploaded_file_regr)
            df_cols = [i for i in df.columns]
            st.dataframe(df)
        rcol1,rcol2,rcol3,rcol4 = st.columns(4)

        with rcol1:
            btn = st.button("show cloumn names for DF.")
            if btn:
                st.write(df.columns)
            
        with rcol2:
            btn = st.button("show all null value.")
            if btn:
                st.write(df.isnull().sum())
            
        with rcol3:
            btn = st.button("Describe DF.")
            if btn:
                st.write(df.describe())
            
        with rcol4:
            show_type_of_data = st.button("Show Datatype.")
            if show_type_of_data:
                st.write(df.dtypes)
        
        #preprocessing
        st.header("PREPROCESS , TRAIN  AND PREDICT")
        st.write("works for only numeric features")
        

        rform = st.form(key='regression')
        select_feature = rform.multiselect("select feature  for Feature Engineering", df_cols)
        target = rform.selectbox("select Target ",df_cols)
        prediction_data = rform.text_input("enter prediction data  (seperator = ','  and no gaps)")

        train = rform.form_submit_button(" Preprocess and train regr ")        
        if train:
            if df is not None:
                df_process = df
            df_process = df_process.dropna()        
            
            if select_feature is not None:
                df_process = df_process[select_feature]
            else:
                df_process = df_process
            
            result_col1,result_col2,result_col3 = st.columns(3)
            with result_col1:
                st.write(df_process)
            with result_col2:
                st.write(f"null values:",df_process.isnull().sum())

            y = df_process[[target]]
            x = df_process.drop([target],axis =1)
            with result_col3:
                st.write("y columns: ",y.columns)
                st.write(f"x columns",x.columns)
            
            trainx , testx, trainy, testy = train_test_split(x,y,test_size=0.2,random_state=42)

            regr = LinearRegression()
            lr_model = regr.fit(trainx,trainy)

            pred = lr_model.predict(testx)

            prediction_data = prediction_data.split(",")
            prediction_data = [float(i) for i in prediction_data]
            prediction_data_result = lr_model.predict([prediction_data])

            st.title(f" predicted column {target} : for data : {prediction_data} is {prediction_data_result}")

            
            scores = {
            'MSE_error' : mean_squared_error(testy,pred),
            'MAE_error' : mean_absolute_error(testy,pred),
             
            }
            st.write(scores)
        
        pass
    pass

if nav_selection=="Description":
    st.header("Project Description")
    st.write(f"""
    Introducing IceAge: Your Ultimate NLP and Machine Learning Solution

IceAge is a revolutionary project that brings together the power of Natural Language Processing (NLP) and machine learning, allowing you to seamlessly perform classification and regression tasks on any dataset. Designed for data scientists, researchers, and professionals alike, IceAge empowers you to unlock valuable insights and make data-driven decisions with ease.

With IceAge's robust classification and regression capabilities, you can tackle a wide range of machine learning challenges. Whether you're predicting customer behavior, sentiment analysis, or financial trends, IceAge equips you with the tools to build accurate and reliable predictive models.

Gone are the days of grappling with complex data preprocessing and feature engineering. IceAge handles all the heavy lifting, automating data preprocessing, feature extraction, and model training. This allows you to focus on the most critical aspects of your analysis, interpretation, and decision-making.

But IceAge doesn't stop at machine learning. It seamlessly integrates powerful NLP functionalities such as sentiment analysis, text summarization, Named Entity Recognition (NER), and POS tagging. This comprehensive NLP toolkit provides you with unparalleled capabilities to extract meaningful insights from text data.

IceAge boasts a user-friendly interface built on the popular Streamlit framework, making it accessible and intuitive for users of all experience levels. Simply upload your dataset, select the desired classification or regression task, and let IceAge work its magic.

Discover the full potential of your data with IceAge. Harness the power of NLP and machine learning to solve complex problems, gain deeper understanding, and make informed decisions. Whether you're in academia, research, or industry, IceAge is your trusted companion on the journey to unlock the true value of your data.





    """)
    pass
