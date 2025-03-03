# Animal Sentiment on Social Media
This project aimed to employ various fundamental knowledge of data wrangling and machine learning to sentiment analysis on social media's textual data.

The data used in this project were obtained through scraping Facebook and Instagram's posts and comments centered around select animals between 2019 and 2023. As emojis (e.g., 😊 😏) and emoticons (e.g., :) <3 ) have become increasingly prevalent to express more nuanced emotions in virtual conversations, this project therefore has taken into account these components (text, emojis, and emoticons) to distill conclusive sentiment within individual expressions.

The overall approach was to apply a Weak Supervision method, emcompassing both unsupervised (e.g., Vader) and supervised techniques (e.g., SVM), to uncover the portrayal of the wildlife by prominent news outlets in Australia, the public opinions around the wildlife, as well as the association between them in shaping online perceptions.

Because we worked with an untouched dataset which posed a variety of issues like spelling errors, abbreviation, mixed expression of text and emojis, a large amount of time has been devoted to preprocessing to prepare clean and tidy data for subsequent sentiment detection.

As suggested by previous research, the overall sentiment of a single document was determined by the following rule:
- If both text and emoji have the same sentiment tag (positive/ neutral/ negative), then the overall sentiment hold that same value.
- If 1 of 2 components (text and emojis) holds neutral sentiment, then the overall sentiment inherits the sentiment tag of the other component.
- If the sentiments of the 2 components are at opposite ends (i.e., one is negative and the other is positive), then the overall sentiment is neutral.

For a quick view, a presentation file is provided in which we summarised this approach from initiation to delivery, together with perceived limitations for upcoming improvement.
