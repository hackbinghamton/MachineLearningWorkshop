import pandas as pd
import datetime
from textstat.textstat import textstat

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import tensorflow as tf

import language_check


def scale_creation_dates(fake_news_df):
    now_timestamp = datetime.datetime.now().timestamp()
    fake_news_df["creation_date"] = fake_news_df["creation_date"].apply(lambda x: x / now_timestamp)

    return fake_news_df


def is_suspicious_country(country):
    sus_countries = ["MK", "PA"]

    return int(country in sus_countries or "REDACTED" in country)


def add_suspicious_country_column(fake_news_df):
    fake_news_df["is_suspicious_country"] = fake_news_df["country"].apply(lambda x: is_suspicious_country(x))

    return fake_news_df


def add_flesch_reading_ease_column(fake_news_df):
    fake_news_df["flesch_reading_ease"] = fake_news_df["article_text"].apply(
        lambda x: (textstat.flesch_reading_ease(x))
    )

    return fake_news_df


def percent_difficult_words(article):
    if textstat.lexicon_count(article) == 0:
        return 0

    return textstat.difficult_words(article) / textstat.lexicon_count(article)


def add_percent_difficult_words_column(fake_news_df):
    fake_news_df["percent_difficult_words"] = fake_news_df["article_text"].apply(lambda x: percent_difficult_words(x))

    return fake_news_df


def typos_to_words(language_tool, article):
    if textstat.lexicon_count(article) == 0:
        return 0

    return len(language_tool.check(article)) / textstat.lexicon_count(article)


def add_percent_typos_to_words_column(fake_news_df):
    language_tool = language_check.LanguageTool("en-US")

    fake_news_df["percent_typos_to_words"] = fake_news_df["article_text"].apply(
        lambda x: typos_to_words(language_tool, x)
    )

    return fake_news_df


def load_fake_news_data(file_name):
    fake_news_data = pd.read_json(file_name)

    fake_news_features = fake_news_data.drop(columns=["is_fake"])
    fake_news_labels = fake_news_data["is_fake"]

    return fake_news_features, fake_news_labels


def load_fake_news_training_data():
    return load_fake_news_data("fakenewsnet_modified_training_set.json")


def load_fake_news_testing_data():
    return load_fake_news_data("fakenewsnet_modified_testing_set.json")


def add_features(fake_news_df):
    fake_news_df = add_suspicious_country_column(fake_news_df)
    fake_news_df = add_flesch_reading_ease_column(fake_news_df)
    fake_news_df = add_percent_difficult_words_column(fake_news_df)
    # fake_news_df = add_percent_typos_to_words_column(fake_news_df)
    
    return fake_news_df


def drop_features(fake_news_df):
    # Drop features we're not using for our machine learning algorithm
    fake_news_df = fake_news_df.drop(columns=["id", "article_text", "country", "title", "news_url"])
    fake_news_df = fake_news_df.reset_index(drop=True)

    return fake_news_df


def refine_fake_news_data(fake_news_df):
    fake_news_df = add_features(fake_news_df)
    fake_news_df = drop_features(fake_news_df)

    return fake_news_df


def evaluate_sklearn_models(training_features, training_labels):
    models = [
        ("Logistic Regression", LogisticRegression(solver="lbfgs")),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Support Vector Machine", SVC(gamma="scale")),
        ("Bagging Classifier", BaggingClassifier()),
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=100))
    ]

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)

        cv_results = model_selection.cross_val_score(
            model, training_features, training_labels, cv=kfold, scoring="accuracy"
        )

        msg = "%s: \n\tAverage accuracy: %f \n\tStandard deviation: %f" % (
            name, cv_results.mean() * 100, cv_results.std() * 100
        )

        print(msg)


def create_neural_network():
    dense_relu_layer = tf.keras.layers.Dense(256, activation="relu")
    dropout_layer = tf.keras.layers.Dropout(0.2)
    dense_softmax_layer = tf.keras.layers.Dense(2, activation="softmax")

    neural_network_model = tf.keras.models.Sequential([
        dense_relu_layer,
        dropout_layer,
        dense_softmax_layer
    ])

    neural_network_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return neural_network_model


def train_neural_network(neural_network_model, training_features, training_labels):
    neural_network_model.fit(training_features.values, training_labels.values, epochs=300)

    return neural_network_model


def evaluate_neural_network(neural_network_model, testing_features, testing_labels):
    test_loss, test_acc = neural_network_model.evaluate(testing_features.values, testing_labels.values)

    return test_acc


def main():
    fake_news_training_features, fake_news_training_labels = load_fake_news_training_data()
    fake_news_testing_features, fake_news_testing_labels = load_fake_news_testing_data()

    fake_news_training_features = refine_fake_news_data(fake_news_training_features)
    fake_news_testing_features = refine_fake_news_data(fake_news_testing_features)

    fake_news_training_features, fake_news_training_labels = fake_news_training_features
    fake_news_testing_features, fake_news_testing_labels = fake_news_testing_features

    evaluate_sklearn_models(fake_news_training_features, fake_news_training_labels)

    neural_network_model = create_neural_network()
    neural_network_model = train_neural_network(
        neural_network_model, fake_news_training_features, fake_news_training_labels
    )

    print(evaluate_neural_network(neural_network_model, fake_news_testing_features, fake_news_testing_labels))


main()
