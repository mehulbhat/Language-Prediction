At the end of program the collowing lines are written:

text = 'okrem iného ako durič na brlohárenie'
text = preprocess_function(text)
text = [split_into_subwords_function(text)]
text_vectorized = vectorizer.transform(text)
model.predict(text_vectorized)

This format can be used to predict any sample data you input as 'text' (as shown above) and you'll get the output as language predicted.
