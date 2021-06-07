import pandas as pd
import numpy as np
import pickle
import streamlit as st

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


# defining the function
def prediction(pclass, sex, age, sibsb, parch, fare, embark, deck):
    prediction = classifier.predict([[pclass, sex, age, sibsb, parch, fare, embark, deck]])
    print(prediction)
    return prediction


# this is the main function in which we define our app
def main():
    st.title("Titanic Chances")

    pclass = st.selectbox('Passenger Class', ('1', '2', '3'))
    st.write('Passenger Class:', pclass)

    sex = st.radio('Gender', ['Male', 'Female'])
    if sex == 'Male':
        sex = 0
    else:
        sex = 1

    age = st.slider('Age')
    st.write('Age:', age)

    sibsb = st.slider('No. of Siblings or Spouses Onboard', 0, 10)
    st.write('No. of Siblings or Spouses Onboard:', sibsb)

    parch = st.slider('No. of Parents or Children Onboard', 0, 10)
    st.write('No. of Parents or Children Onboard:', parch)

    fare = st.slider('Ticket Fare', 0, 500)
    st.write('Ticket Fare:', fare)

    embark = st.selectbox('Port of Embarkation', ('Cherbourg', 'Queenstown', 'Southampton'))
    st.write('Port of Embarkation:', embark)
    if sex == 'Cherbourg':
        embark = 0
    elif sex == 'Queenstown':
        embark = 1
    else:
        embark =2

    deck = st.selectbox('Residence Deck', ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'))
    st.write('Residence Deck:', deck)
    if deck == 'A':
        deck = 0
    elif deck == 'B':
        deck = 1
    elif deck == 'C':
        deck = 2
    elif deck == 'D':
        deck = 3
    elif deck == 'E':
        deck = 4
    elif deck == 'F':
        deck = 5
    elif deck == 'G':
        deck = 6
    elif deck == 'T':
        deck = 7
    else:
        deck = 8


    result = ""


    # the below line ensures that when the button called 'GO' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("GO"):
        result = prediction(pclass, sex, age, sibsb, parch, fare, embark, deck)
        if result == 1:
            st.subheader('Passenger would have survived :smile:')
        else:
            st.subheader('Passenger would not have survived :cry:')





if __name__ == '__main__':
    main()
