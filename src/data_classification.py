# classifies the reviews into three categories (good, neutral, bad)

import pandas as pd


# sets a label based on the review score
def review_classification(score):
    if score >= 4:
        return 'good'
    elif score == 3:
        return 'neutral'
    else:
        return 'bad'


if __name__ == '__main__':
    # read data
    df = pd.read_csv('./data/Reviews.csv')

    # set labels
    df['label'] = df['Score'].apply(review_classification)

    # save new dataset with label
    df.to_csv('./data/reviews_with_labels.csv', index=False)
