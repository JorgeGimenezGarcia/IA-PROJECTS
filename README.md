# AI PROJECTS

## Overview

In this project, I created a logistic regression model that predicts whether a user's review will be positive or negative.

### How does the AI learn to classify reviews?
- I used the IMDb dataset, which provides detailed data to differentiate between positive and negative reviews.

### How is the review analysis performed?
- I utilized the `sklearn` library, specifically the `HashingVectorizer`. This library helps transform words into vectors, allowing the model to perform cross-validation between positive and negative reviews.

### Special Note:
For this project, I built a Flask application to visually deploy the model, making it easier for users to interact with and understand the code.

---

## Setup Instructions

To activate the environment and run the app:

```bash
source env/bin/activate
python3 app.py


This project implements a logistic regression model that predicts whether a user’s review is positive or negative.

Folder Structure
aclImdb: Contains test and train folders for training and testing datasets.
pkl_objects: Stores serialized pkl files for the trained logistic regression model.
env: Includes the virtual environment and a Flask application for interacting with the model.
static and templates: Flask's static and template files, including:
_formerhelpers.html: Displays guidance if invalid data is entered.
resultados.html: Shows model prediction results.
reviewformulario.html: The main form for submitting reviews.
Core Files
app.py: Runs the Flask app and connects to the SQL database to store and use prediction data.
update.py: Updates the pkl objects and database with new data.
vectorizer.py: Cleans user reviews (e.g., removes emojis, punctuation) for consistency.
Libraries Used
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import sqlite3
import numpy as np
from vectorizer import vect
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
----
Large Movie Review Dataset v1.0

Overview

This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification. This document outlines how the dataset was
gathered, and how to use the files provided. 

Dataset 

The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg). We also include an additional 50,000 unlabeled
documents for unsupervised learning. 

In the entire collection, no more than 30 reviews are allowed for any
given movie because reviews for the same movie tend to have correlated
ratings. Further, the train and test sets contain a disjoint set of
movies, so no significant performance is obtained by memorizing
movie-unique terms and their associated with observed labels.  In the
labeled train/test sets, a negative review has a score <= 4 out of 10,
and a positive review has a score >= 7 out of 10. Thus reviews with
more neutral ratings are not included in the train/test sets. In the
unsupervised set, reviews of any rating are included and there are an
even number of reviews > 5 and <= 5.

Files

There are two top-level directories [train/, test/] corresponding to
the training and test sets. Each contains [pos/, neg/] directories for
the reviews with binary labels positive and negative. Within these
directories, reviews are stored in text files named following the
convention [[id]_[rating].txt] where [id] is a unique id and [rating] is
the star rating for that review on a 1-10 scale. For example, the file
[test/pos/200_8.txt] is the text for a positive-labeled test set
example with unique id 200 and star rating 8/10 from IMDb. The
[train/unsup/] directory has 0 for all ratings because the ratings are
omitted for this portion of the dataset.

We also include the IMDb URLs for each review in a separate
[urls_[pos, neg, unsup].txt] file. A review with unique id 200 will
have its URL on line 200 of this file. Due the ever-changing IMDb, we
are unable to link directly to the review, but only to the movie's
review page.

In addition to the review text files, we include already-tokenized bag
of words (BoW) features that were used in our experiments. These 
are stored in .feat files in the train/test directories. Each .feat
file is in LIBSVM format, an ascii sparse-vector format for labeled
data.  The feature indices in these files start from 0, and the text
tokens corresponding to a feature index is found in [imdb.vocab]. So a
line with 0:7 in a .feat file means the first word in [imdb.vocab]
(the) appears 7 times in that review.

LIBSVM page for details on .feat file format:
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

We also include [imdbEr.txt] which contains the expected rating for
each token in [imdb.vocab] as computed by (Potts, 2011). The expected
rating is a good way to get a sense for the average polarity of a word
in the dataset.

Citing the dataset

When using this dataset please cite our ACL 2011 paper which
introduces it. This paper also contains classification results which
you may want to compare against.


@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

References

Potts, Christopher. 2011. On the negativity of negation. In Nan Li and
David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20,
636-659.

Contact

For questions/comments/corrections please contact Andrew Maas
amaas@cs.stanford.edu
