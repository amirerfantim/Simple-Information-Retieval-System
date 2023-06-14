import json
import re
import math
from collections import Counter
import hazm
from hazm import *

try:
    with open("../IR_data_news_12k.json") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: File not found.")
    exit()
except:
    print("Error: Failed to load news data from file.")
    exit()


def persian_sort_key(t):
    return [ord(c) for c in list(t[0])]


def phase1():
    contents = [data[str(i)]["content"] for i in range(len(data))]
    urls = [data[str(i)]["url"] for i in range(len(data))]
    titles = [data[str(i)]["title"] for i in range(len(data))]

    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>'
        , '?', '@', '[', ']', '^', '_', '`', '،', '{', '|', '}',
                    '~', '«', '»', '…', 'ْ', 'ٌ', 'ٍ', 'ً', 'ُ', 'ِ', 'َ', 'ّ', 'ء', 'ٔ', 'ٰ', '﷼']

    all_tokens = []
    stops = set(stopwords_list())
    stemmer = Stemmer()
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    for c in contents:
        c = normalizer.normalize(c)
        tokens = word_tokenize(c)

        for i in range(len(tokens)):
            tokens[i] = (tokens[i], i)
        filtered_tokens = []
        for i in range(len(tokens)):
            if tokens[i][0] not in stops:
                if tokens[i][0] not in punctuations:
                    filtered_tokens.append(tokens[i])
        tokens = filtered_tokens

        stemmed_tokens = []
        for i in range(len(tokens)):
            stemmed_token = stemmer.stem(tokens[i][0])
            stemmed_tokens.append((stemmed_token, tokens[i][1]))
        tokens = stemmed_tokens

        lemmatized_tokens = []
        for i in range(len(tokens)):
            lemma = lemmatizer.lemmatize(tokens[i][0])
            lemmatized_tokens.append((lemma, tokens[i][1]))
        tokens = lemmatized_tokens

        all_tokens.append(tokens)
    return all_tokens


# postings list
# for each token -> (doc_id, pos), doc_freq, total_freq
def create_positional_indexes(ppd):
    postings_lists = {}
    for doc_id, doc in enumerate(ppd):
        process_document(postings_lists, doc_id, doc)
    return postings_lists


def process_document(postings_list, doc_id, doc):
    for token, pos in doc:
        if token not in postings_list:
            postings_list[token] = create_new_term_entry()
        if doc_id in postings_list[token]['posting']:
            postings_list[token]['posting'][doc_id].append(pos)
        else:
            postings_list[token]['posting'][doc_id] = [pos]
            postings_list[token]['doc_frequency'] += 1
        postings_list[token]['total_frequency'] += 1


def calculate_term_frequencies(doc):
    term_freq = {}
    for token, pos in doc:
        if token not in term_freq:
            term_freq[token] = 0
        term_freq[token] += 1
    return term_freq


# def calculate_tf_idf(postings_list, doc_id, ppd, doc):
#     term_freq = calculate_term_frequencies(doc)
#     for token in term_freq:
#         tf = math.log2(term_freq[token]) + 1
#         idf = math.log2(len(ppd) / postings_list[token]['doc_frequency'])
#         tf_idf = tf * idf
#         postings_list[token]['tf_idf'][doc_id] = tf_idf


# def update_postings_lists(postings_lists, ppd):
#     for doc_id, doc in enumerate(ppd):
#         calculate_tf_idf(postings_lists, doc_id, ppd, doc)


def calculate_tf_idf(posting_lists, ppd):
    document_length = {}
    for token in posting_lists.keys():
        for doc_id in posting_lists[token]['posting'].keys():
            term_freq = len(posting_lists[token]['posting'][doc_id])
            tf = math.log2(term_freq) + 1
            idf = math.log2(len(ppd) / posting_lists[token]['doc_frequency'])
            tf_idf = tf * idf
            if doc_id in document_length.keys():
                document_length[doc_id] += tf_idf ** 2
            else:
                document_length[doc_id] = tf_idf ** 2
            posting_lists[token]['tf_idf'][doc_id] = tf_idf
    for doc in document_length.keys():
        document_length[doc] = math.sqrt(document_length[doc])
    return document_length


def create_new_term_entry():
    new_entry = {}
    new_entry['posting'] = {}
    new_entry['total_frequency'] = 0
    new_entry['doc_frequency'] = 0
    new_entry['tf_idf'] = {}
    return new_entry


ppd = phase1()
positional_indexes_lists = create_positional_indexes(ppd)


# update_postings_lists(positional_indexes_lists, ppd)


def tokenize_query(query):
    # replace quoted substrings with placeholders
    placeholders = []

    def repl(m):
        placeholders.append(m.group())
        return f'___{len(placeholders) - 1}___'

    query = re.sub(r'"[^"]*"', repl, query)

    # split by spaces
    tokens = query.split()

    # replace placeholders with original substrings
    for i, ph in enumerate(placeholders):
        tokens = [t.replace(f'___{i}___', ph) for t in tokens]

    return tokens


def rank_documents(ppq, posting_lists):
    rank_docs = {}
    pos_docs = {}
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    normalizer = Normalizer()

    for token in ppq:
        if token.startswith('!'):
            token = token[1:]
            token = stemmer.stem(token)
            token = lemmatizer.lemmatize(token)
            doc_ids = set(posting_lists[token]['posting'].keys())
            for doc_id in doc_ids:
                rank_docs[doc_id] = -math.inf
        elif token.startswith('"'):
            multi_word_token = token[1:-1]
            multi_word_token = normalizer.normalize(multi_word_token)
            multi_word_token = multi_word_token.split()
            docs_containing_all_parts = set()
            for i, part in enumerate(multi_word_token):
                part = stemmer.stem(part)
                part = lemmatizer.lemmatize(part)
                if i == 0:
                    docs_containing_all_parts = set(posting_lists[part]['posting'].keys())
                else:
                    doc_ids = set(posting_lists[part]['posting'].keys())
                    docs_containing_all_parts = docs_containing_all_parts.intersection(doc_ids)
            for doc_id in docs_containing_all_parts:
                positions = []
                added = False
                for tok in multi_word_token:
                    tok = stemmer.stem(tok)
                    tok = lemmatizer.lemmatize(tok)
                    positions.append(posting_lists[tok]['posting'][doc_id])
                for i in range(len(positions[0])):
                    flag = True
                    for j in range(len(multi_word_token)):
                        if positions[0][i] + j not in positions[j]:
                            flag = False
                            break
                    if doc_id not in pos_docs and flag:
                        pos_docs[doc_id] = []
                    if flag:
                        pos_docs[doc_id].append(positions[0][i])
                    if flag and not added:
                        added = True
                        if doc_id not in rank_docs:
                            rank_docs[doc_id] = 0
                        rank_docs[doc_id] += 1
        else:
            token = stemmer.stem(token)
            token = lemmatizer.lemmatize(token)
            doc_ids = set(posting_lists[token]['posting'].keys())
            for doc_id in doc_ids:
                if doc_id not in rank_docs:
                    rank_docs[doc_id] = 0
                rank_docs[doc_id] += 1
                if doc_id not in pos_docs:
                    pos_docs[doc_id] = []
                for pos in posting_lists[token]['posting'][doc_id]:
                    pos_docs[doc_id].append(pos)

    sorted_ranks = sorted(rank_docs.items(), key=lambda x: x[1], reverse=True)
    return sorted_ranks, pos_docs


def get_result(ranked, positions):
    for i in range(5):
        doc_id = ranked[i][0]
        print()
        print('Document ID: ' + str(doc_id))
        print('Title: ' + data[str(doc_id)]['title'])
        content = data[str(doc_id)]['content']
        normalizer = Normalizer()
        content = normalizer.normalize(content)
        content = word_tokenize(content)
        counter = 1
        for pos in positions[doc_id]:
            main_pos = pos
            pos_to_print = []
            j = -1
            while main_pos + j > 0 and content[main_pos + j] != '.':
                pos_to_print.insert(0, main_pos + j)
                j -= 1
            pos_to_print.append(main_pos)
            i = 1
            while main_pos + i < len(content) and content[main_pos + i] != '.':
                pos_to_print.append(main_pos + i)
                i += 1
            sentence = ''
            for word in pos_to_print:
                sentence += content[word]
                sentence += ' '
            print('Sentence ' + str(counter) + ': ' + sentence)
            counter += 1


# Example usage:

# query = 'مایکل !جردن'
# ppq = tokenize_query(query)


# ranked_docs, position_of_tokens = rank_documents(ppq, positional_indexes_lists)
# get_result(ranked_docs, position_of_tokens)


# PHASE 2.1
doc_len = calculate_tf_idf(positional_indexes_lists, ppd)


def show_posting_list(word, postings_lists):
    if word in postings_lists:
        posting_list = postings_lists[word]['posting']
        tf_idf_list = postings_lists[word]['tf_idf']
        print(f"Posting list for '{word}':")
        for doc_id, positions in posting_list.items():
            tf_idf = tf_idf_list[doc_id]
            print(f"Document ID: {doc_id}")
            print(f"Positions: {positions}")
            print(f"TF-IDF: {tf_idf}")
            print("---")
    else:
        print(f"No posting list found for '{word}'.")
    print(postings_lists['سجاد']['doc_frequency'])


# Example usage:
word_to_search = "سجاد"

# show_posting_list(word_to_search, positional_indexes_lists)
query = "ماریو گومز"


# PHASE 2.2.1
def calculate_tf_idf_sum(doc_id, positional_indexes_lists):
    tf_idf_sum = 0.0

    for token in positional_indexes_lists.keys():
        if doc_id in positional_indexes_lists[token]['tf_idf']:
            tf_idf_sum += pow(positional_indexes_lists[token]['tf_idf'][doc_id], 2)

    return math.sqrt(tf_idf_sum)


def calculate_cosine_similarity(query, positional_indexes_lists):
    query_tf, query_sum = calculate_query_tf(query)
    doc_cosine_score_dic = calculate_doc_cosine_scores(query_tf, positional_indexes_lists)
    doc_cosine_similarity = calculate_normalized_cosine_similarity(doc_cosine_score_dic, positional_indexes_lists,
                                                                   query_sum)
    sorted_doc_cosine_similarity = sort_cosine_similarity(doc_cosine_similarity)
    return sorted_doc_cosine_similarity


def calculate_query_tf(query):
    query_tf = {}

    for token in tokenize_query(query):
        if token in query_tf.keys():
            query_tf[token] += 1
        else:
            query_tf[token] = 1

    for token in query_tf.keys():
        query_tf[token] = 1 + math.log2(query_tf[token])

    tf_sum = sum(math.pow(tf, 2) for tf in query_tf.values())

    return query_tf, math.sqrt(tf_sum)


def calculate_doc_cosine_scores(query_tf, positional_indexes_lists):
    doc_cosine_score_dic = {}

    for token in query_tf.keys():
        if token in positional_indexes_lists.keys():
            for doc_id in positional_indexes_lists[token]['posting'].keys():
                if doc_id in doc_cosine_score_dic.keys():
                    doc_cosine_score_dic[doc_id] += query_tf[token] * positional_indexes_lists[token]['tf_idf'][doc_id]
                else:
                    doc_cosine_score_dic[doc_id] = query_tf[token] * positional_indexes_lists[token]['tf_idf'][doc_id]

    return doc_cosine_score_dic


def calculate_normalized_cosine_similarity(doc_cosine_score_dic, positional_indexes_lists, query_sum):
    doc_cosine_similarity = {}

    for doc_id in doc_cosine_score_dic.keys():
        doc_cosine_similarity[doc_id] = doc_cosine_score_dic[doc_id] / doc_len[doc_id]
        doc_cosine_similarity[doc_id] = doc_cosine_similarity[doc_id] / query_sum

    return doc_cosine_similarity


def sort_cosine_similarity(doc_cosine_similarity):
    sorted_doc_cosine_similarity = sorted(doc_cosine_similarity.items(), key=lambda x: x[1], reverse=True)
    return sorted_doc_cosine_similarity


# def calculate_query_tf_sum(query):
#     query_tf = {}
#
#     for token in tokenize_query(query):
#         if token in query_tf:
#             query_tf[token] += 1
#         else:
#             query_tf[token] = 1
#     for token in query_tf.keys():
#         query_tf[token] = 1 + math.log2(query_tf[token])
#
#     tf_sum = sum(math.pow(tf, 2) for tf in query_tf.values())
#     return tf_sum

cosine_similarity = calculate_cosine_similarity(query, positional_indexes_lists)


# PHASE 2.2.2
def calculate_jaccard_score(query, positional_indexes_lists):
    query_tokens = set(tokenize_query(query))
    doc_jaccard_scores = {}
    for term in query_tokens:
        if term in positional_indexes_lists:
            for doc_id in positional_indexes_lists[term]['posting'].keys():
                if doc_id not in doc_jaccard_scores.keys():
                    doc_tokens = set([token for token, _ in ppd[doc_id]])
                    intersection = len(query_tokens.intersection(doc_tokens))
                    union = len(query_tokens.union(doc_tokens))
                    jaccard_score = intersection / union
                    doc_jaccard_scores[doc_id] = jaccard_score

    sorted_doc_jaccard_scores = sorted(doc_jaccard_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_doc_jaccard_scores


jaccard_scores = calculate_jaccard_score(query, positional_indexes_lists)


# PHASE 3
def create_champion_lists(postings_lists, k):
    champion_lists = {}
    for term in postings_lists:
        tf_idf_scores = postings_lists[term]['tf_idf']
        tf_idf_scores = {}
        for doc_id in postings_lists[term]['posting'].keys():
            if doc_id in tf_idf_scores.keys():
                tf_idf_scores[doc_id] += postings_lists[term]['tf_idf'][doc_id] / doc_len[doc_id]
            else:
                tf_idf_scores[doc_id] = postings_lists[term]['tf_idf'][doc_id] / doc_len[doc_id]

        sorted_docs = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
        r = min(k * 3, len(sorted_docs))
        champion_lists[term] = [doc_id for doc_id, _ in
                                sorted_docs[:r]]

    return champion_lists


def calculate_cosine_similarity_with_champion_lists(query, positional_indexes_lists, champion_lists):
    query_tf, query_sum = calculate_query_tf(query)
    doc_cosine_score_dic = calculate_doc_cosine_scores_with_champion_lists(query_tf, positional_indexes_lists,
                                                                           champion_lists)
    doc_cosine_similarity = calculate_normalized_cosine_similarity(doc_cosine_score_dic, positional_indexes_lists,
                                                                   query_sum)
    sorted_doc_cosine_similarity = sort_cosine_similarity(doc_cosine_similarity)
    return sorted_doc_cosine_similarity


def calculate_doc_cosine_scores_with_champion_lists(query_tf, positional_indexes_lists, champion_lists):
    doc_cosine_score_dic = {}

    for token in query_tf.keys():
        if token in positional_indexes_lists.keys():
            if token in champion_lists.keys():
                for doc_id in champion_lists[token]:
                    if doc_id in doc_cosine_score_dic.keys():
                        doc_cosine_score_dic[doc_id] += query_tf[token] * positional_indexes_lists[token]['tf_idf'][
                            doc_id]
                    else:
                        doc_cosine_score_dic[doc_id] = query_tf[token] * positional_indexes_lists[token]['tf_idf'][
                            doc_id]

    return doc_cosine_score_dic


def calculate_jaccard_score_champion(query, champion_lists, positional_indexes_lists):
    query_tokens = set(tokenize_query(query))
    doc_jaccard_scores = {}

    for term in query_tokens:
        if term in champion_lists:
            for doc_id in champion_lists[term]:
                if doc_id not in doc_jaccard_scores.keys():
                    doc_tokens = set([token for token, _ in ppd[doc_id]])
                    intersection = len(query_tokens.intersection(doc_tokens))
                    union = len(query_tokens.union(doc_tokens))
                    jaccard_score = intersection / union
                    doc_jaccard_scores[doc_id] = jaccard_score

    sorted_doc_jaccard_scores = sorted(doc_jaccard_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_doc_jaccard_scores


def print_result(scores, k, name):
    i = 0
    for doc_id, score in scores:
        if i == k:
            break
        i += 1
        print(f"Document ID: {doc_id}")
        print(f"{name} Score: {score}")
        print(f"Title: {data[str(doc_id)]['title']}")
        print("---")


k = 20
s = 5
champion_lists = create_champion_lists(positional_indexes_lists, k)
jaccard_scores_champion = calculate_jaccard_score_champion(query, champion_lists, positional_indexes_lists)
cosine_similarity_with_champion_lists = calculate_cosine_similarity_with_champion_lists(query, positional_indexes_lists,
                                                                                        champion_lists)


# print(tokenize_query(query))
#
# print_result(jaccard_scores_champion, s, "jaccard Champion:")
# print()
# print('***************************************')
# print()
# print_result(jaccard_scores, s, "jaccard:")
# print()
# print('***************************************')
# print()
# print_result(cosine_similarity_with_champion_lists, s, "cosine champion:")
# print()
# print('***************************************')
# print()
# print_result(cosine_similarity, s, "cosine:")


def print_res(scores, query, postings_lists, s):
    x = 0
    for doc_id, score in scores:
        if x == s:
            break
        x += 1
        print()
        print('Document ID: ' + str(doc_id))
        print('Title: ' + data[str(doc_id)]['title'])
        print('Score: ', score)
        content = data[str(doc_id)]['content']
        normalizer = Normalizer()
        content = normalizer.normalize(content)
        content = word_tokenize(content)
        counter = 1
        tokenized = tokenize_query(query)
        for token in tokenized:
            if token not in postings_lists:
                continue
            if doc_id not in postings_lists[token]['posting']:
                continue
            for pos in postings_lists[token]['posting'][doc_id]:
                main_pos = pos
                pos_to_print = []
                j = -1
                while main_pos + j > 0 and content[main_pos + j] != '.':
                    pos_to_print.insert(0, main_pos + j)
                    j -= 1
                pos_to_print.append(main_pos)
                i = 1
                while main_pos + i < len(content) and content[main_pos + i] != '.':
                    pos_to_print.append(main_pos + i)
                    i += 1
                sentence = ''
                for word in pos_to_print:
                    sentence += content[word]
                    sentence += ' '
                print('Sentence ' + str(counter) + ': ' + sentence)
                counter += 1


print_res(cosine_similarity_with_champion_lists, query, positional_indexes_lists, s)
print()
print("***************")
print()
print_res(jaccard_scores_champion, query, positional_indexes_lists, s)

