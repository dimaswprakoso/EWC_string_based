import mysql.connector
import datetime
import time
import result_log as logging

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import spatial
from collections import OrderedDict
from operator import itemgetter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import array
from numpy.linalg import norm

import math as math

starttime = datetime.datetime.now()
print("start  :%s" % starttime)

# ----------------------------------------------------------------------------#
# Configuration
# ----------------------------------------------------------------------------#
db_user = 'root'
db_database = 'sharebox'
language = 'EN'

# testing---------------------------------------------------------------------#
method = 'string' # string-based
sent_sim = 'string'
ic = 'no' # information content
top_n_list = [10] # N-value in Top-N recommendation

string_sim = 'string'
word_sim_algo = 'string'
base_word = 'raw'  # pre-processing, options: raw, stem, lemma

min_sim = 0.000001  # recommendation threshold
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
# 0. Initialize
# ----------------------------------------------------------------------------#
# Mysql Connection
cnx = mysql.connector.connect(user=db_user, database=db_database)
cursor = cnx.cursor(dictionary=True)

# ----------------------------------------------------------------------------#
# 1. Get items from workshop
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `workshop_items2` WHERE language = '""" + language + """' AND type='Material' 
"""

# RTRIM(LTRIM(wastecode)) != '99 99 99' LIMIT 0,25

item_list = {}
try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        item_list[row['id']] = [row['Waste_description'], row['Wastecode']]

    print("Items rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

# ----------------------------------------------------------------------------#
# 2. Get items from ewc
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `ewc_level3`
"""

ewc_list = {}

try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        ewc_list[row['EWC_level3']] = [row['description'], row['id']]

    print("EWC rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))


# ----------------------------------------------------------------------------#
# Similarity Functions
# ----------------------------------------------------------------------------#
def NLP(data):
    # 0. Lowercase
    # global NLP_counter
    # NLP_counter += 1

    data = data.lower()

    # 1. Tokenize # word tokenize (removes also punctuation)
    tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
    words = tokenizer.tokenize(data)

    # 2. Remove short words
    words = [w for w in words if len(w) > 2]

    # 3. Remove Stopwords # load stopwords from enlgihs language
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]  # for each word check if

    # 4 Remove common terminology in waste listings e.g. (waste)

    term_list = ['waste', 'scrap', 'scraps', 'process', 'processes', 'processed', 'processing', 'unprocessed',
                 'consultancy', 'advice', 'training', 'service', 'managing', 'management', 'recycling', 'recycle',
                 'industry', 'industrial', 'material', 'materials', 'quantity', 'support', 'residue', 'organic',
                 'remainder', 'specific', 'particular', 'solution', 'solutions', 'substance', 'substances', 'product',
                 'production', 'use', 'used', 'unused', 'consumption', 'otherwise', 'specified', 'based', 'spent',
                 'hazardous', 'dangerous', 'containing', 'other']

    words = [w for w in words if not w in term_list]  # for each word check if
    data = words

    # 5. Find Stem/Lemma

    if base_word == 'stem':
        ps = PorterStemmer()
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        data = stemmed_words
    elif base_word == 'lemma':
        lm = WordNetLemmatizer()
        lemmatized_words = []
        for w in words:
            lemmatized_words.append(lm.lemmatize(w))
        data = lemmatized_words

    return data


def cos_similarity(item_vec1, item_vec2):
    sim = 1 - spatial.distance.cosine(item_vec1, item_vec2)  # cosine similarity

    return sim


def find_unique_words(data, data2):
    # merge words from all items, then remove duplicates with set datatype
    all_unique_words = []
    all_unique_words = list(set(all_unique_words + data))

    for i, j in data2.items():
        all_unique_words = list(set(all_unique_words + j))

    return all_unique_words


def gen_item_vector(data, all_unique_words):
    # Using all words in dataset create a vector format,
    # then for each item in the dataset, initialize that vector with direction

    # create a list of item vectors, initialize each item vector with zero values
    vec = {}
    vec = [0] * len(all_unique_words)

    # Initialize each item vector
    for m in data:  # for each word
        # Update the word-index of the item vector
        vec[all_unique_words.index(m)] += 1

    # return list of item vectors
    return vec


# ----------------------------------------------------------------------------#
# 3. Recommendation
# ----------------------------------------------------------------------------#
def recommend(item_desc, ewc_words):
    item_vec1 = {}
    item_vec2 = {}
    sim_list = {}
    idf = {}

    ## uw = find_unique_words(item_desc, ewc_words)
    it = 0

    # match the words from the item description against the words of each EWC code description
    for k, l in ewc_words.items():
        # Lets do some matching -->
        uw = find_unique_words(item_desc, {k: l})
        # uw = ['steel','paper','metal','iron']
        item_vec1[it] = gen_item_vector(l, uw)  # ewc code vector
        item_vec2[it] = gen_item_vector(item_desc, uw)  # item desc vector

        # TF-IDF weighting------------------------------------------------------
        # calculate idf for each vocab and multiply it with vector element
        i = 0
        for vocab in uw:
            N = 2
            df = 0
            if vocab in l:
                df += 1
            if vocab in item_desc:
                df += 1
            # idf = math.log10((1 + N) / (1 + df)) + 1
            idf = math.log((1 + N) / (1 + df)) + 1

            # multiply it right away to the vector
            item_vec1[it][i] = item_vec1[it][i] * idf
            item_vec2[it][i] = item_vec2[it][i] * idf

            i += 1

        # calculate L2-Norm or Euclidean distance
        vec1_norm = norm(array(item_vec1[it]))
        vec2_norm = norm(array(item_vec2[it]))

        # normalize the vectors
        for vector_index in range(i):
            item_vec1[it][vector_index] = item_vec1[it][vector_index] / vec1_norm
            item_vec2[it][vector_index] = item_vec2[it][vector_index] / vec2_norm

        # TF-IDF weighting end ---------------------------------------------------

        # check if item vector is not empty
        if sum(item_vec1[it]) > 0 and sum(item_vec2[it]) > 0:
            sim_list[k] = cos_similarity(item_vec1[it], item_vec2[it])

        it += 1

    return sim_list


def generate_recommendation_list(sim_matrix, top_n, min_sim):
    # default value
    if top_n is None:
        top_n = 1

    rec_list = {}
    top_itter = 0

    s = [(k, sim_matrix[k]) for k in sorted(sim_matrix, key=sim_matrix.get, reverse=True)]
    for k, v in s:
        # top 10 and similarity is high enough
        if top_itter < top_n and v > min_sim:
            rec_list[k] = v
        top_itter += 1

    rec_list = OrderedDict(sorted(rec_list.items(), key=itemgetter(1), reverse=True))

    return rec_list


# ----------------------------------------------------------------------------#
# 4. Evaluation
# ----------------------------------------------------------------------------#
def eval_topn(rec_list, ewc):
    it = 1
    m = {}
    m['no_rec'] = len(rec_list)  # how many recommendations were provided
    m['correct'] = 0  # was the right recommendation in the list
    m['position'] = 0  # What was the position (no 2 out of 10) of the right
    m['ewc_label'] = 1  # Some items have '99 99 99', thus no EWC code assigned. Needed for EWC
    m['rhr'] = 0  # reciprocal hit-rank

    if ewc == '99 99 99':
        m['ewc_label'] = 0

    for i, j in rec_list.items():
        # print(i+" -<>- "+ewc)

        if i == ewc:
            m['correct'] = 1
            m['position'] = it
            m['rhr'] = 1 / it
        # if ewc == '99 99 99':
        #     m['ewc_label'] = 0

        it += 1

    return m


def eval_recommendations(ev):
    m = {}  # dictionary having all performance metrics

    m['no_items'] = len(ev)  # number of items for which recommendation could be provided

    m['no_rec'] = 0  # total recommendations provided
    for i, j in ev.items():
        if j['no_rec'] > 0:
            m['no_rec'] += 1

    m['no_labeled'] = 0
    for i, j in ev.items():
        if j['ewc_label'] == 1:
            m['no_labeled'] += 1

    m['tp'] = 0  # True positives (inherent to all correct recommendations)
    for i, j in ev.items():
        # if j['correct'] > 0:
        if j['no_rec'] > 0 and j['correct'] > 0:
            m['tp'] += 1

    m['fp'] = 0  # False positives (inherent to incorrect recommendations)
    for i, j in ev.items():
        # if j['correct'] == 0:
        if j['no_rec'] > 0 and j['correct'] == 0:
            m['fp'] += 1

    m['tn'] = 0  # True negatives ()
    for i, j in ev.items():
        # if j['ewc_label'] == 1 and j['no_rec'] == 0:
        if j['no_rec'] == 0 and j['ewc_label'] == 0:
            m['tn'] += 1

    m['fn'] = 0  # False negatives ()
    for i, j in ev.items():
        # if j['correct'] == 0 and j['no_rec'] == 0:
        if j['no_rec'] == 0 and j['ewc_label'] == 1:
            m['fn'] += 1

    rhr_total = 0
    for i, j in ev.items():
        if j['correct'] > 0:
            rhr_total += j['rhr']

    # --------------------------------- #

    # Precision = TP / TP + FP
    if m['tp'] + m['fp'] > 0:
        m['precision'] = m['tp'] / (m['tp'] + m['fp'])
    else:
        m['precision'] = 0

    # Recall = TP / no_items_with_ewc_label
    if m['no_labeled'] > 0:
        m['recall'] = m['tp'] / m['no_labeled']
    else:
        m['recall'] = 0

    # Accuracy = TP + True Negatives / all items
    if m['no_items'] > 0:
        m['accuracy'] = (m['tp'] + m['tn']) / m['no_items']
    else:
        m['accuracy'] = 0

    # F1 measure
    if m['precision'] + m['recall'] > 0:
        m['f1'] = 2 * ((m['precision'] * m['recall']) / (m['precision'] + m['recall']))
    else:
        m['f1'] = 0

    # Average Reciprocal Hit-rank
    m['arhr'] = rhr_total / len(ev)

    # Return all measures
    return m


# ----------------------------------------------------------------------------#
# 5. Main code
# ----------------------------------------------------------------------------#
# --------------- Config -------------------- #
ev = {}  # dictionary containing all evaluations of recommendations
ewc_words = {}  # bag of words from the ewc description
item_words = {}  # bag of words from the item description
rec = {}  # dictionary containing the recommendations for an item desc
sim_matrix = {}  # the similarity matrix between the vectors of ewc desc and item desc
# --------------- End Config ---------------- #


# Prepare the item_desc
for i, j in item_list.items():
    # clean EWC data
    item_ewc = j[1].strip()

    # delete * symbol from ewc code
    # translation_table = dict.fromkeys(map(ord, '*'), None)
    # item_ewc = item_ewc.translate(translation_table)
    item_list[i] = [j[0], item_ewc.strip()]

    desc = j[0].strip()

    # Generate word list for each waste description
    item_words[i] = NLP(desc)

# prepare the ewc_desc
for k, l in ewc_list.items():
    ewc_words[k] = NLP(l[0])


# conduct series of test
for top_n in top_n_list:
    start = time.time()

    # for all items
    for m, n in item_list.items():
        # Generate the similarity matrix
        sim_matrix = recommend(item_words[m], ewc_words)

        # generate the list of recommendations for an item description
        rec = generate_recommendation_list(sim_matrix, top_n, min_sim)

        # Evaluate if the recommendation was correct (with stats)
        ev[m] = eval_topn(rec, item_list[m][1])

    # Calculate the performance metrics (e.g. recall, ARHR, precision) over all items
    print(ev)
    # logging.log_result_ev(ev)
    results = eval_recommendations(ev)
    print(results)

    end = time.time()
    log = {'duration': end - start, 'method': method, 'sent_sim': sent_sim, 'ic': ic, 'string_sim': string_sim,
           'word_sim_algo': word_sim_algo, 'base_word': base_word, 'pos': 'all', 'word_sim_th': 1, 'top_n': top_n}
    log.update(results)
    logging.log_result(log)

endtime = datetime.datetime.now()
print("start  :%s" % starttime)
print("end    :%s" % endtime)
print("elapsed:%s" % (endtime - starttime))
