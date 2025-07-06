from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


def get_vectorizer():
    get_title = FunctionTransformer(lambda x: x["title"], validate=False)
    get_text = FunctionTransformer(lambda x: x["full_text"], validate=False)

    vectorizer = FeatureUnion(
        [
            (
                "title",
                Pipeline(
                    [
                        ("selector", get_title),
                        (
                            "tfidf",
                            TfidfVectorizer(ngram_range=(1, 2), max_features=3000),
                        ),
                    ]
                ),
            ),
            (
                "full_text",
                Pipeline(
                    [
                        ("selector", get_text),
                        (
                            "tfidf",
                            TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
                        ),
                    ]
                ),
            ),
        ]
    )
    return vectorizer
