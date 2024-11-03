import pickle
import csv
import os
import random

import torch
from torch.utils.data import dataset

from util import *
from run import *

'''
def _get_table_dict(tables):
    table_dict = {}
    for t in tables:
        split = t.split(' ')
        if len(split) > 1:
            # Alias -> full table name.
            table_dict[split[1]] = split[0]
        #else:
            # Just full table name.
        table_dict[split[0]] = split[0]
    return table_dict
'''

def load_samples(file_name, num_queries, num_materialized_samples):
    samples = []
    print(file_name)
    print(num_queries)
    print(num_materialized_samples)
    if os.path.isfile(file_name) and num_materialized_samples > 0:
        print("Sample bitmaps file exists")
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open(file_name, 'rb') as f:
            for i in range(num_queries):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")
    else:
        print("Sample bitmaps file not exists")
        samples = [[]] * num_queries
    return samples

def load_query(csv_file, templates={}, query_num=0, contain_labels=True):
    tables = []
    joins = []
    predicates = []
    #columns = []
    #operators = []
    #values = []
    queries = []
    template_distribution = {t:0 for t in templates}
    template_labels = []
    query_labels = []
    with open (csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for i, row in enumerate(data_raw):
            if query_num > 0 and i >= query_num:
                break
            table = row[0].split(',')
            join = row[1].split(',')
            predicate = row[2].split(',')
            if contain_labels:
                truecard = int(row[3])
            else:
                truecard = 0
            if truecard < 1:
                truecard = 1
            query_labels.append(truecard)
            #print(truecard)
            queries.append([table, join, predicate])
    #true_qnum = len(queries)
    #while len(queries) < query_num:
    #    for i in range(true_qnum):
    #        if len(queries) < query_num:
    #            queries.append(queries[i])
    #            query_labels.append(query_labels[i])
    normalized_queries = []
    for lineid, query in enumerate(queries):
        if lineid % 1000 == 0:
            print(lineid)
        table = query[0]
        #print(table)
        join = query[1]
        predicate = query[2]
        #table_dict = _get_table_dict(table)
        # Add all table names into tables.
        t_fullname = [] 
        for t in table:
            if len(t.split(' ')) > 1:
                t_fullname.append(t.split(' ')[1])
            else:
                t_fullname.append(t)
        t_fullname.sort()
        tables.append(t_fullname)
        #for t_alias in table_dict:
        #    if table_dict[t_alias] not in tables:
        #        tables.append(table_dict[t_alias])
        # Add all joins. Only consider equal join. Replace table alias to table name.
        # The join tables are in alphabet order to avoid duplicate templates.
        q_joins = []
        for j in join:
            if len(j) <= 1:
                q_joins.append('')
                continue
            j_t1 = j.split('=')[0]
            j_t2 = j.split('=')[1]
            #t1_alias = j_t1.split('.')[0]
            #t1_column = j_t1.split('.')[1]
            #t2_alias = j_t2.split('.')[0]
            #t2_column = j_t2.split('.')[1]
            #t1 = table_dict[t1_alias]
            #t2 = table_dict[t2_alias]
            # sort the lvalue and rvalue or equal join
            j_list = [j_t1, j_t2]
            j_list.sort()
            normalized_j = j_list[0] + '=' + j_list[1]
            if normalized_j not in joins:
                q_joins.append(normalized_j)
        # sort all joins
        q_joins.sort()
        joins.append(q_joins)
        # Add all predicates. 
        q_cols = []
        q_ops = []
        q_vals = []
        q_predicates = []
        template_predicates = []
        for i  in range(0, len(predicate), 3):
            if len(predicate) < 3:
                q_predicates.append([])
                break
            col = predicate[i]
            op = predicate[i + 1]
            val = predicate[i + 2]
            t1 = col.split('.')[0]
            #col_name = col.split('.')[1]
            #t1 = table_dict[t_alias]
            #full_col = '{}.{}'.format(t1, col_name)
            #q_cols.append(full_col)
            #q_ops.append(op)
            #q_vals.append(val)
            template_predicates.append(col)
            q_predicates.append([col, op, val])
        template_predicates.sort()
        q_template = ','.join(t_fullname) + '#' + ','.join(q_joins) + '#' + ','.join(template_predicates)
        #print(q_template)
        if q_template not in templates:
            templates[q_template] = len(templates)
            template_distribution[q_template] = 1
        else:
            template_distribution[q_template] += 1
        template_labels.append(templates[q_template])
        predicates.append(q_predicates)
    print("Number of templates: ", len(templates))
    return templates, template_distribution, tables, joins, predicates, query_labels, template_labels

def generate_queries_with_template(templates, template_distribution, col_min_max_vals, total_num = 50000):
    total_distribution = 0
    for t in template_distribution:
        total_distribution += template_distribution[t]
    #total_distribution = np.sum(template_distribution)
    tables = []
    joins = []
    predicates = []
    template_labels = []
    for t in templates:
        gen_num = int(total_num * template_distribution[t] / total_distribution)
        q_tables = t.split('#')[0].split(',')
        q_joins = t.split('#')[1].split(',')
        cols = t.split('#')[2].split(',')
        ops = []
        vals = []
        for col in cols:
            ops.append(np.random.choice(['<', '>', '='], size=gen_num))
            vals.append(np.random.choice(np.arange(start=col_min_max_vals[col][0], stop=col_min_max_vals[col][1])))
        for i in range(gen_num):
            tables.append(q_tables)
            joins.append(q_joins)
            q_ops = [ops[j][i] for j in range(len(cols))]
            q_vals = [vals[j][i] for j in range(len(cols))]
            predicates.append([cols, ops, vals])
            template_labels.append(templates[t])
    
    return tables, joins, predicates, template_labels


def generate_non_label_training_queries(templates, template_distribution, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, total_num, samples):
    tables, joins, predicates, template_labels = generate_queries_with_template(templates, template_distribution, column_min_max_vals, total_num)
    cols_enc, opvals_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)

    num_queries = len(tables)

    order = np.arange(num_queries)
    random.shuffle(order)
    cols = [cols_enc[i] for i in order]
    opvals = [opvals_enc[i] for i in order]
    joins = [joins_enc[i] for i in order]

    max_num_joins = max([len(j) for j in joins])
    max_num_cols = max([len(p) for p in cols])
    max_num_opvals = max([len(p) for p in opvals])

    train_data = [samples, cols, opvals, joins]
    train_labels = [0 for _ in range(num_queries)]

    return train_data, train_labels, template_labels 


'''Currently Used'''
def generate_positive_queries(data, idxdicts, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, template_labels):
    #samples, cols, opvals, joins = data
    samples, predicates, joins = data
    vec2table, vec2column, vec2op, vec2join = idxdicts
    #newopvals = []
    newpredicates = []
    #for opval in opvals:
    for predicate in predicates:
        #newopval = []
        newpredicate = []
        #for ov in opval:
        for cov in predicate:
            col = cov[:-len(vec2op) - 1]
            op = cov[-len(vec2op) - 1:-1]
            val = cov[-1]
            opidx = np.where(op == 1)[0]
            if len(opidx) == 0:
                newcov = cov 
            elif vec2op[opidx[0]] != '=':
                choice = random.randint(0, 1)
                bias = random.uniform(0, 0.1)
                if choice == 0:
                    newval = max(0, val - bias)
                else:
                    newval = min(1, val + bias) 
                newcov = np.hstack((col, op, [newval]))
            else:
                newcov = cov 
            #newopval.append(newov)
            newpredicate.append(newcov)
        #newopvals.append(newopval)
        newpredicates.append(newpredicates)
    #newdata = [samples, cols, newopvals, joins]
    newdata = [samples, predicates, joins]
    labels = [0 for _ in range(len(samples))]
    return newdata, labels, template_labels





    

'Currently Used'
def encode_data_from_saver(query_file, sample_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, conn=None, cursor=None, model=None, num_samples=0, query_num=0):
    templates, template_distribution, tables, joins, predicates, query_labels, template_labels = load_query(query_file, templates=templates, query_num=query_num)
    samples = load_samples(sample_bitmap_file, len(tables), num_materialized_samples=num_samples)
    if len(tables) < query_num:
        while len(tables) < query_num:
            for i in range(len(tables)):
                if len(tables) < query_num:
                    tables.append(tables[i])
                    joins.append(joins[i])
                    predicates.append(predicates[i])
                    query_labels.append(query_labels[i])
                    template_labels.append(template_labels[i])
                    samples.append(samples[i])
    samples_enc = encode_samples(tables, samples, table2vec)
    cols_enc, opvals_enc, joins_enc = encode_col_opval_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    predicates_enc, joins_enc = encode_predicate_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, _, _ = normalize_labels(query_labels, min_val=min_val, max_val=max_val)

    num_queries = len(tables)
    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    #cols_train = cols_enc[:num_train]
    #opvals_train = opvals_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]
    template_labels_train = template_labels[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    #cols_test = cols_enc[num_train:num_train + num_test]
    #opvals_test = opvals_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]
    template_labels_test = template_labels[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    #max_num_cols = max(max([len(p) for p in cols_train]), max([len(p) for p in cols_test]))
    #max_num_opvals = max(max([len(p) for p in opvals_train]), max([len(p) for p in opvals_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    #dicts = [table2vec, column2vec, op2vec, join2vec]
    #train_data = [samples_train, cols_train, opvals_train, joins_train]
    #test_data = [samples_test, cols_test, opvals_test, joins_test]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    #return labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_data, test_data, template_labels_train, template_labels_test, templates, template_distribution
    return labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data, template_labels_train, template_labels_test, templates, template_distribution


'''Currently used'''
def templatize_and_encode_data(query_file, sample_bitmaps_file, adaptation_query_file, adaptation_sample_bitmaps_file, min_max_file, num_samples=0, repeat=10):
    # Read and analyze queries and templates from file.
    templates, template_distribution, source_tables, source_joins, source_predicates, source_query_labels, source_template_labels = load_query(query_file)
    source_samples = load_samples(sample_bitmaps_file, len(source_tables), num_materialized_samples=num_samples)
    templates, template_distribution, target_tables, target_joins, target_predicates, target_query_labels, target_template_labels = load_query(adaptation_query_file, templates=templates)
    adaptation_samples = load_samples(adaptation_sample_bitmaps_file, len(target_tables), num_materialized_samples=num_samples)
    print("Finish load training queries.")
    #columns = predicates[0]
    #operators = predicates[1]
    #values = predicates[2]
    samples = source_samples + adaptation_samples
    tables =source_tables + target_tables
    joins = source_joins + target_joins
    predicates = source_predicates + target_predicates
    query_labels = source_query_labels + target_query_labels
    template_labels = source_template_labels + target_template_labels

    for i in range(repeat):
        samples += adaptation_samples
        tables += target_tables
        joins += target_joins
        predicates += target_predicates
        query_labels += target_query_labels
        template_labels += target_template_labels

    # Get distinct names of tables, joins, columns and operators
    #table_names = get_all_names(tables)
    #join_names = get_all_names(joins)
    column_names, operators = get_all_names_predicates(predicates)
    if "=" not in operators:
        operators.append("=")
    #operators = get_all_name(operators)
    
    # Encoding all distinct names as onehot encoding
    #table2vec, idx2table = get_set_encoding(table_names)
    #join2vec, idx2join = get_set_encoding(join_names)
    #column2vec, idx2column = get_set_encoding(column_names)
    op2vec, idx2op = get_set_encoding(operators)

    
    column_min_max_vals = {}
    all_column_names = []
    with open(min_max_file + "column_min_max_vals.csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]
            all_column_names.append(row[0])
    column2vec, idx2column = get_set_encoding(all_column_names)

    all_table_names = []
    with open(min_max_file + "tables.csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            all_table_names.append(row[0])
    table2vec, idx2table = get_set_encoding(all_table_names)
    
    all_join_names = []
    with open(min_max_file + "joins.csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            all_join_names.append(row[0])
    all_join_names.append("")
    join2vec, idx2join = get_set_encoding(all_join_names)

    '''
    positive_tables = []
    positive_joins = []
    positive_predicates = []
    for i in range(len(tables)):
        table = tables[i]
        join = joins[i]
        predicate = predicates[i]
        new_table, new_join, new_predicate = generate_with_template([table, join, predicate], column_min_max_vals, 10)
        positive_tables.append(new_table)
        positive_joins.append(new_join)
        positive_predicates.append(new_predicate)
    '''

    samples_enc = encode_samples(tables, samples, table2vec)
    #cols_enc, opvals_enc, joins_enc = encode_col_opval_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    predicates_enc, joins_enc = encode_predicate_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(query_labels)

    '''
    positive_cols_enc = []
    positive_joins_enc = []
    positive_opvals_enc = []
    for i in range(len(positive_joins)):
        col_enc, opval_enc, join_enc = encode_data(positive_predicates[i], positive_joins[i], column_min_max_vals, column2vec, op2vec, join2vec)
        positive_cols_enc.append(col_enc)
        positive_joins_enc.append(join_enc)
        positive_opvals_enc.append(opval_enc)
    '''

    num_queries = len(tables)
    # Split in training and validation samples
    num_test = len(target_query_labels)
    num_train = num_queries - num_test
    #num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    #cols_train = cols_enc[:num_train]
    #opvals_train = opvals_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]
    template_labels_train = template_labels[:num_train]
    '''
    positive_cols_train = positive_cols_enc[:num_train]
    positive_joins_train = positive_joins_enc[:num_train]
    positive_opvals_train = positive_opvals_enc[:num_train]
    '''

    samples_test = samples_enc[num_train:num_train + num_test]
    #cols_test = cols_enc[num_train:num_train + num_test]
    #opvals_test = opvals_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]
    template_labels_test = template_labels[num_train:num_train + num_test]
    '''
    positive_cols_test = positive_cols_enc[num_train:num_train + num_test]
    positive_joins_test = positive_joins_enc[num_train:num_train + num_test]
    positive_opvals_test = positive_opvals_enc[num_train:num_train + num_test]
    '''

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    #max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    #max_num_cols = max(max([len(p) for p in cols_train]), max([len(p) for p in cols_test]))
    #max_num_opvals = max(max([len(p) for p in cols_train]), max([len(p) for p in opvals_test]))
    #max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))
    max_num_joins = len(all_join_names)
    max_num_predicates = 58 

    dicts = [table2vec, column2vec, op2vec, join2vec]
    idxdicts = [idx2table, idx2column, idx2op, idx2join]
    #train_data = [samples_train, cols_train, opvals_train, joins_train, positive_cols_train, positive_joins_train, positive_opvals_train]
    #test_data = [samples_test, cols_test, opvals_test, joins_test, positive_cols_test, positive_joins_test, positive_opvals_test]
    #train_data = [samples_train, cols_train, opvals_train, joins_train]
    #test_data = [samples_test, cols_test, opvals_test, joins_test]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    return dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, \
                max_num_predicates, train_data, test_data, templates, template_distribution, template_labels_train, template_labels_test
                #max_num_cols, max_num_opvals, train_data, test_data, templates, template_distribution, template_labels_train, template_labels_test

def make_triplet_dataset(anchor_data, anchor_labels, anchor_template_labels, positive_data, positive_labels, positive_template_labels, negative_data, \
                         negative_labels, negative_template_labels, max_num_joins, max_num_cols, max_num_opvals):
    anchor_samples, anchor_cols, anchor_opvals, anchor_joins = anchor_data
    positive_samples, positive_cols, positive_opvals, positive_joins = positive_data
    negative_samples, negative_cols, negative_opvals, negative_joins = negative_data

    all_sample_masks = []
    all_sample_tensors = []
    for samples in [anchor_samples, positive_samples, negative_samples]:
        sample_masks = []
        sample_tensors = []
        if len(samples) == 0:
            continue
        for sample in samples:
            sample_tensor = np.vstack(sample)
            num_pad = max_num_joins + 1 - sample_tensor.shape[0]
            sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
            sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
            sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
            sample_tensors.append(np.expand_dims(sample_tensor, 0))
            sample_masks.append(np.expand_dims(sample_mask, 0))
        sample_tensors = np.vstack(sample_tensors)
        sample_tensors = torch.FloatTensor(sample_tensors)
        sample_masks = np.vstack(sample_masks)
        sample_masks = torch.FloatTensor(sample_masks)
        all_sample_masks.append(sample_masks)
        all_sample_tensors.append(sample_tensors)

    all_col_masks = []
    all_col_tensors = []
    for cols in [anchor_cols, positive_cols, negative_cols]:
        col_masks = []
        col_tensors = []
        if len(cols) == 0:
            continue
        for col in cols:
            col_tensor = np.vstack(col)
            num_pad = max_num_cols - col_tensor.shape[0]
            col_mask = np.ones_like(col_tensor).mean(1, keepdims=True)
            col_tensor = np.pad(col_tensor, ((0, num_pad), (0, 0)), 'constant')
            col_mask = np.pad(col_mask, ((0, num_pad), (0, 0)), 'constant')
            col_tensors.append(np.expand_dims(col_tensor, 0))
            col_masks.append(np.expand_dims(col_mask, 0))
        col_tensors = np.vstack(col_tensors)
        col_tensors = torch.FloatTensor(col_tensors)
        col_masks = np.vstack(col_masks)
        col_masks = torch.FloatTensor(col_masks)
        all_col_masks.append(col_masks)
        all_col_tensors.append(col_tensors)
    
    all_opval_masks = []
    all_opval_tensors = []
    for i, opvals in enumerate([anchor_opvals, positive_opvals, negative_opvals]):
        opval_masks = []
        opval_tensors = []
        if len(opvals) == 0:
            continue
        for opval in opvals:
            opval_tensor = np.vstack(opval)
            num_pad = max_num_opvals - opval_tensor.shape[0]
            opval_mask = np.ones_like(opval_tensor).mean(1, keepdims=True)
            opval_tensor = np.pad(opval_tensor, ((0, num_pad), (0, 0)), 'constant')
            opval_mask = np.pad(opval_mask, ((0, num_pad), (0, 0)), 'constant')
            opval_tensors.append(np.expand_dims(opval_tensor, 0))
            opval_masks.append(np.expand_dims(opval_mask, 0))
        opval_tensors = np.vstack(opval_tensors)
        opval_tensors = torch.FloatTensor(opval_tensors)
        opval_masks = np.vstack(opval_masks)
        opval_masks = torch.FloatTensor(opval_masks)
        all_opval_masks.append(opval_masks)
        all_opval_tensors.append(opval_tensors)

    all_join_masks = []
    all_join_tensors = []
    for joins in [anchor_joins, positive_joins, negative_joins]:
        join_masks = []
        join_tensors = []
        if len(joins) == 0:
            continue
        for join in joins:
            join_tensor = np.vstack(join)
            num_pad = max_num_joins - join_tensor.shape[0]
            join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
            join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
            join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
            join_tensors.append(np.expand_dims(join_tensor, 0))
            join_masks.append(np.expand_dims(join_mask, 0))
        join_tensors = np.vstack(join_tensors)
        join_tensors = torch.FloatTensor(join_tensors)
        join_masks = np.vstack(join_masks)
        join_masks = torch.FloatTensor(join_masks)
        all_join_masks.append(join_masks)
        all_join_tensors.append(join_tensors)

    #anchor_data = [all_sample_tensors[0], all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0]]
    #positive_data = [all_sample_tensors[1], all_col_tensors[1], all_opval_tensors[1], all_join_tensors[1]]
    #negative_data = [all_sample_tensors[2], all_col_tensors[2], all_opval_tensors[2], all_join_tensors[2]]

    #anchor_masks = [all_sample_masks[0], all_col_masks[0], all_opval_masks[0], all_join_masks[0]]
    #positive_masks = [all_sample_masks[1], all_col_masks[1], all_opval_masks[1], all_join_masks[1]]
    #negative_masks = [all_sample_masks[2], all_col_masks[2], all_opval_masks[2], all_join_masks[2]]
    anchor_labels = torch.FloatTensor(anchor_labels)
    positive_labels = torch.FloatTensor(positive_labels)
    negative_labels = torch.FloatTensor(negative_labels)
    anchor_template_labels = torch.FloatTensor(anchor_template_labels)
    positive_template_labels = torch.FloatTensor(positive_template_labels)
    negative_template_labels = torch.FloatTensor(negative_template_labels)

    print(sample_tensors.size())
    print(col_tensors.size())
    print(opval_tensors.size())
    print(join_tensors.size())
    print(sample_masks.size())
    print(col_masks.size())
    print(opval_masks.size())
    print(join_masks.size())
    #print(labels_tensor.size())

    return dataset.TensorDataset(all_sample_tensors[0], all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0], \
                                 all_sample_masks[0], all_col_masks[0], all_opval_masks[0], all_join_masks[0], anchor_labels, anchor_template_labels, \
                                 all_sample_tensors[1], all_col_tensors[1], all_opval_tensors[1], all_join_tensors[1], \
                                 all_sample_masks[1], all_col_masks[1], all_opval_masks[1], all_join_masks[1], positive_labels, positive_template_labels, \
                                 all_sample_tensors[2], all_col_tensors[2], all_opval_tensors[2], all_join_tensors[2], \
                                 all_sample_masks[2], all_col_masks[2], all_opval_masks[2], all_join_masks[2], negative_labels, negative_template_labels)    

'''Currently used'''
#def make_tensors(data, max_num_joins, max_num_cols, max_num_opvals):
def make_tensors(data, max_num_joins, max_num_predicates):
    #samples, cols, opvals, joins = data
    samples, predicates, joins = data
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    #sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    #sample_masks = torch.FloatTensor(sample_masks)
    
    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    #col_tensors = torch.FloatTensor(col_tensors)
    predicate_masks = np.vstack(predicate_masks)
    #col_masks = torch.FloatTensor(col_masks)

    '''
    col_masks = []
    col_tensors = []
    for col in cols:
        col_tensor = np.vstack(col)
        num_pad = max_num_cols - col_tensor.shape[0]
        col_mask = np.ones_like(col_tensor).mean(1, keepdims=True)
        col_tensor = np.pad(col_tensor, ((0, num_pad), (0, 0)), 'constant')
        col_mask = np.pad(col_mask, ((0, num_pad), (0, 0)), 'constant')
        col_tensors.append(np.expand_dims(col_tensor, 0))
        col_masks.append(np.expand_dims(col_mask, 0))
    col_tensors = np.vstack(col_tensors)
    #col_tensors = torch.FloatTensor(col_tensors)
    col_masks = np.vstack(col_masks)
    #col_masks = torch.FloatTensor(col_masks)

    opval_masks = []
    opval_tensors = []
    for opval in opvals:
        opval_tensor = np.vstack(opval)
        num_pad = max_num_opvals - opval_tensor.shape[0]
        opval_mask = np.ones_like(opval_tensor).mean(1, keepdims=True)
        opval_tensor = np.pad(opval_tensor, ((0, num_pad), (0, 0)), 'constant')
        opval_mask = np.pad(opval_mask, ((0, num_pad), (0, 0)), 'constant')
        opval_tensors.append(np.expand_dims(opval_tensor, 0))
        opval_masks.append(np.expand_dims(opval_mask, 0))
    opval_tensors = np.vstack(opval_tensors)
    #opval_tensors = torch.FloatTensor(opval_tensors)
    opval_masks = np.vstack(opval_masks)
    #opval_masks = torch.FloatTensor(opval_masks)
    '''

    join_masks = [] 
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    #join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    #join_masks = torch.FloatTensor(join_masks)

    #labels_tensor = torch.FloatTensor(labels)
    #return [sample_tensors, col_tensors, opval_tensors, join_tensors, sample_masks, col_masks, opval_masks, join_masks]
    return [sample_tensors, predicate_tensors, join_tensors, sample_masks, predicate_masks, join_masks]

"Currently Used"
def make_single_dataset(data, max_num_joins, max_num_predicates, labels):
    #samples, cols, opvals, joins = data
    samples, predicates, joins = data
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)
    
    '''
    col_masks = []
    col_tensors = []
    for col in cols:
        col_tensor = np.vstack(col)
        num_pad = max_num_cols - col_tensor.shape[0]
        col_mask = np.ones_like(col_tensor).mean(1, keepdims=True)
        col_tensor = np.pad(col_tensor, ((0, num_pad), (0, 0)), 'constant')
        col_mask = np.pad(col_mask, ((0, num_pad), (0, 0)), 'constant')
        col_tensors.append(np.expand_dims(col_tensor, 0))
        col_masks.append(np.expand_dims(col_mask, 0))
    col_tensors = np.vstack(col_tensors)
    col_tensors = torch.FloatTensor(col_tensors)
    col_masks = np.vstack(col_masks)
    col_masks = torch.FloatTensor(col_masks)


    opval_masks = []
    opval_tensors = []
    for opval in opvals:
        opval_tensor = np.vstack(opval)
        num_pad = max_num_opvals - opval_tensor.shape[0]
        opval_mask = np.ones_like(opval_tensor).mean(1, keepdims=True)
        opval_tensor = np.pad(opval_tensor, ((0, num_pad), (0, 0)), 'constant')
        opval_mask = np.pad(opval_mask, ((0, num_pad), (0, 0)), 'constant')
        opval_tensors.append(np.expand_dims(opval_tensor, 0))
        opval_masks.append(np.expand_dims(opval_mask, 0))
    opval_tensors = np.vstack(opval_tensors)
    opval_tensors = torch.FloatTensor(opval_tensors)
    opval_masks = np.vstack(opval_masks)
    opval_masks = torch.FloatTensor(opval_masks)
    '''

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    labels_tensor = torch.FloatTensor(labels)


    #return dataset.TensorDataset(sample_tensors, col_tensors, opval_tensors, join_tensors, labels_tensor, sample_masks, col_masks, opval_masks, join_masks)
    return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, labels_tensor, sample_masks, predicate_masks, join_masks)

'''Currently Used'''
def make_dataset(source_samples, source_predicates, source_joins, source_labels, max_num_joins, max_num_predicates, source_template_labels,
                target_samples=[], target_predicates=[], target_joins=[], target_labels=[],  target_template_labels=[]):
    """Add zero-padding and wrap as tensor dataset."""
    all_sample_masks = []
    all_sample_tensors = []
    for samples in [source_samples, target_samples]:
        sample_masks = []
        sample_tensors = []
        if len(samples) == 0:
            continue
        for sample in samples:
            sample_tensor = np.vstack(sample)
            num_pad = max_num_joins + 1 - sample_tensor.shape[0]
            sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
            sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
            sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
            sample_tensors.append(np.expand_dims(sample_tensor, 0))
            sample_masks.append(np.expand_dims(sample_mask, 0))
        sample_tensors = np.vstack(sample_tensors)
        sample_tensors = torch.FloatTensor(sample_tensors)
        sample_masks = np.vstack(sample_masks)
        sample_masks = torch.FloatTensor(sample_masks)
        all_sample_masks.append(sample_masks)
        all_sample_tensors.append(sample_tensors)

    all_predicate_masks = []
    all_predicate_tensors = []
    for predicates in [source_predicates, target_predicates]:
        predicate_masks = []
        predicate_tensors = []
        if len(predicates) == 0:
            continue
        for predicate in predicates:
            predicate_tensor = np.vstack(predicate)
            num_pad = max_num_predicates - predicate_tensor.shape[0]
            predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
            predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
            predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
            predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
            predicate_masks.append(np.expand_dims(predicate_mask, 0))
        predicate_tensors = np.vstack(predicate_tensors)
        predicate_tensors = torch.FloatTensor(predicate_tensors)
        predicate_masks = np.vstack(predicate_masks)
        predicate_masks = torch.FloatTensor(predicate_masks)
        all_predicate_masks.append(predicate_masks)
        all_predicate_tensors.append(predicate_tensors)

    ''' 
    all_opval_masks = []
    all_opval_tensors = []
    for opvals in [source_opvals, target_opvals]:
        opval_masks = []
        opval_tensors = []
        if len(opvals) == 0:
            continue
        for opval in opvals:
            opval_tensor = np.vstack(opval)
            num_pad = max_num_opvals - opval_tensor.shape[0]
            opval_mask = np.ones_like(opval_tensor).mean(1, keepdims=True)
            opval_tensor = np.pad(opval_tensor, ((0, num_pad), (0, 0)), 'constant')
            opval_mask = np.pad(opval_mask, ((0, num_pad), (0, 0)), 'constant')
            opval_tensors.append(np.expand_dims(opval_tensor, 0))
            opval_masks.append(np.expand_dims(opval_mask, 0))
        opval_tensors = np.vstack(opval_tensors)
        opval_tensors = torch.FloatTensor(opval_tensors)
        opval_masks = np.vstack(opval_masks)
        opval_masks = torch.FloatTensor(opval_masks)
        all_opval_masks.append(opval_masks)
        all_opval_tensors.append(opval_tensors)
    '''

    all_join_masks = []
    all_join_tensors = []
    for joins in [source_joins, target_joins]:
        join_masks = []
        join_tensors = []
        if len(joins) == 0:
            continue
        for join in joins:
            join_tensor = np.vstack(join)
            num_pad = max_num_joins - join_tensor.shape[0]
            join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
            join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
            join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
            join_tensors.append(np.expand_dims(join_tensor, 0))
            join_masks.append(np.expand_dims(join_mask, 0))
        join_tensors = np.vstack(join_tensors)
        join_tensors = torch.FloatTensor(join_tensors)
        join_masks = np.vstack(join_masks)
        join_masks = torch.FloatTensor(join_masks)
        all_join_masks.append(join_masks)
        all_join_tensors.append(join_tensors)

    source_label_tensor = torch.FloatTensor(source_labels)
    source_template_labels_tensor = torch.FloatTensor(source_template_labels)

    if len(target_template_labels) > 0:
        target_label_tensor = torch.FloatTensor(target_labels)
        target_template_labels_tensor = torch.FloatTensor(target_template_labels)
        if len(all_sample_tensors) > 0:
            return dataset.TensorDataset(all_sample_tensors[0], all_predicate_tensors[0], all_join_tensors[0], source_label_tensor,
                                     all_sample_masks[0], all_predicate_masks[0], all_join_masks[0], source_template_labels_tensor,
                                     all_sample_tensors[1], all_predicate_tensors[1], all_join_tensors[1], target_label_tensor,
                                     all_sample_masks[1], all_predicate_masks[1], all_join_masks[1], target_template_labels_tensor)
            #return dataset.TensorDataset(all_sample_tensors[0], all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0], source_label_tensor,
            #                         all_sample_masks[0], all_col_masks[0], all_opval_masks[0], all_join_masks[0], source_template_labels_tensor,
            #                         all_sample_tensors[1], all_col_tensors[1], all_opval_tensors[1], all_join_tensors[1], target_label_tensor,
            #                         all_sample_masks[1], all_col_masks[1], all_opval_masks[1], all_join_masks[1], target_template_labels_tensor)
        else:
            #return dataset.TensorDataset(all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0], source_label_tensor,
            #                         all_col_masks[0], all_opval_masks[0], all_join_masks[0], source_template_labels_tensor,
            #                         all_col_tensors[1], all_opval_tensors[1], all_join_tensors[1], target_label_tensor,
            #                         all_col_masks[1], all_opval_masks[1], all_join_masks[1], target_template_labels_tensor)
            return dataset.TensorDataset(all_predicate_tensors[0], all_join_tensors[0], source_label_tensor,
                                     all_predicate_masks[0], all_join_masks[0], source_template_labels_tensor,
                                     all_predicate_tensors[1], all_join_tensors[1], target_label_tensor,
                                     all_predicate_masks[1], all_join_masks[1], target_template_labels_tensor)
    if len(all_sample_tensors) > 0:
        #return dataset.TensorDataset(all_sample_tensors[0], all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0], source_label_tensor, all_sample_masks[0],
        #                         all_col_masks[0], all_opval_masks[0], all_join_masks[0])
        return dataset.TensorDataset(all_sample_tensors[0], all_predicate_tensors[0], all_join_tensors[0], source_label_tensor, all_sample_masks[0],
                                 all_predicate_masks[0], all_join_masks[0])
    else:
        #return dataset.TensorDataset(all_col_tensors[0], all_opval_tensors[0], all_join_tensors[0], source_label_tensor,
        #                         all_col_masks[0], all_opval_masks[0], all_join_masks[0])
        return dataset.TensorDataset(all_predicate_tensors[0], all_join_tensors[0], source_label_tensor,
                                 all_predicate_masks[0], all_join_masks[0])
        
'''
def get_train_datasets(query_file, sample_bitmap_file, min_max_file, num_samples=0):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_data, \
                test_data, templates, template_distribution, template_labels_train, template_labels_test = templatize_and_encode_data(query_file, sample_bitmap_file, min_max_file, num_samples=num_samples)
    print("Loaded and encoded training data")
    train_dataset = make_dataset(*train_data, source_labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_cols=max_num_cols, max_num_opvals=max_num_opvals, source_template_labels=template_labels_train)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, source_labels=labels_test, max_num_joins=max_num_joins,
                                max_num_cols=max_num_cols, max_num_opvals=max_num_opvals, source_template_labels=template_labels_test)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, \
            train_dataset, test_dataset, templates, template_distribution
'''

'''
def get_train_datasets(query_file, sample_bitmap_file, min_max_file, num_samples=0):
    #dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_data, \
    dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, max_num_opvals, train_data, \
                test_data, templates, template_distribution, template_labels_train, template_labels_test = templatize_and_encode_data(query_file, sample_bitmap_file, min_max_file, num_samples=num_samples)
    #order = np.arange(len(labels_train))
    order = np.argsort(labels_train)
    random.shuffle(order)
    #samples_train, cols_train, opvals_train, joins_train = train_data
    samples_train, predicates_train, joins_train = train_data
    target_samples_train = [samples_train[i] for i in order]
    #target_cols_train = [cols_train[i] for i in order]
    #target_opvals_train = [opvals_train[i] for i in order]
    target_predicates_train = [predicates_train[i] for i in order]
    target_joins_train = [joins_train[i] for i in order]
    target_labels_train = [labels_train[i] for i in order]
    target_template_labels_train = [template_labels_train[i] for i in order]

    order = np.arange(len(labels_test))
    random.shuffle(order)
    #samples_test, cols_test, opvals_test, joins_test = test_data
    samples_test, predicates_test, joins_test = test_data
    target_samples_test = [samples_test[i] for i in order]
    #target_cols_test = [cols_test[i] for i in order]
    #target_opvals_test = [opvals_test[i] for i in order]
    target_predicates_test = [predicates_test[i] for i in order]
    target_joins_test = [joins_test[i] for i in order]
    target_labels_test = [labels_test[i] for i in order]
    target_template_labels_test = [template_labels_test[i] for i in order]

    print("Loaded and encoded training data")
    #train_dataset = make_dataset(*train_data, source_labels=labels_train, max_num_joins=max_num_joins,
    #                             max_num_cols=max_num_cols, max_num_opvals=max_num_opvals, source_template_labels=template_labels_train)
    #train_dataset = make_dataset(train_data[0], train_data[1], train_data[2], train_data[3], labels_train, max_num_joins,
    #                            max_num_cols, max_num_opvals, template_labels_train, target_samples_train, target_cols_train, target_opvals_train, target_joins_train,
    #                            target_labels_train, target_template_labels_train)
    train_dataset = make_dataset(train_data[0], train_data[1], train_data[2], train_data[3], labels_train, max_num_predicates,
                                max_num_opvals, template_labels_train, target_samples_train, target_predicates_train, target_joins_train,
                                target_labels_train, target_template_labels_train)
    print("Created TensorDataset for training data")
    #test_dataset = make_dataset(test_data[0], test_data[1], test_data[2], test_data[3], labels_test, max_num_joins,
    #                            max_num_cols, max_num_opvals, template_labels_test, target_samples_test, target_cols_test, target_opvals_test, target_joins_test,
    #                            target_labels_test, target_template_labels_test)
    test_dataset = make_dataset(test_data[0], test_data[1], test_data[2], test_data[3], labels_test, max_num_predicates,
                                max_num_opvals, template_labels_test, target_samples_test, target_predicates_test, target_joins_test,
                                target_labels_test, target_template_labels_test)
    print("Created TensorDataset for validation data")
    #return dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, target_labels_train, target_labels_test, max_num_joins, max_num_cols, max_num_opvals, \
    #        train_dataset, test_dataset, templates, template_distribution, train_data, test_data
    return dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, target_labels_train, target_labels_test, max_num_predicates, max_num_opvals, \
            train_dataset, test_dataset, templates, template_distribution, train_data, test_data
'''

'''
def get_source_target_datasets(train_query_file, adaptation_query_file, train_bitmap_file, adaptation_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=0, query_num=0):
    source_labels_train, source_labels_test, max_num_joins, max_num_cols, max_num_opvals, source_data_train, source_data_test, source_template_labels_train, source_template_labels_test \
                            =  encode_data_from_saver(train_query_file, train_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)
    target_labels_train, target_labels_test, _, _, _, target_data_train, target_data_test, target_template_labels_train, target_template_labels_test \
                            = encode_data_from_saver(adaptation_query_file, adaptation_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)
    #train_dataset = make_dataset(source_data_train[0], source_data_train[1], source_data_train[2], source_data_train[3], source_labels_train, max_num_joins,
    #                            max_num_cols, max_num_opvals, source_template_labels_train, target_data_train[0], target_data_train[1], target_data_train[2], target_data_train[3],
    #                            target_labels_train, target_template_labels_train)
    #test_dataset = make_dataset(source_data_test[0], source_data_test[1], source_data_test[2], source_data_test[3], source_labels_test, max_num_joins,
    #                            max_num_cols, max_num_opvals, source_template_labels_test, target_data_test[0], target_data_test[1], target_data_test[2], target_data_test[3],
    #                            target_labels_test, target_template_labels_test)
    source_data_train_tensors = make_tensors(source_data_train, max_num_joins, max_num_cols, max_num_opvals)
    source_data_test_tensors = make_tensors(source_data_test, max_num_joins, max_num_cols, max_num_opvals)
    target_data_train_tensors = make_tensors(target_data_train, max_num_joins, max_num_cols, max_num_opvals)
    target_data_test_tensors = make_tensors(target_data_test, max_num_joins, max_num_cols, max_num_opvals)

    return source_labels_train, source_labels_test, target_labels_train, target_labels_test, source_data_train_tensors, source_data_test_tensors, target_data_train_tensors, target_data_test_tensors 
'''


def check_triplet(anchor_data, anchor_labels, positive_data, positive_labels, negative_data, negative_labels):
    for i in range(len(anchor_data)):
        anchor_label = anchor_labels[i]
        positive_label = positive_labels[i] 
        negative_label = negative_labels[i]
        #if abs(anchor_label - positive_label) > abs(anchor_label - negative_label):

def tensor2numpy(data, cuda):
    if cuda == 'cuda':
        data = [l.cpu() for l in data]
    npdata = np.array(data, dtype=np.float32)
    return npdata

def rearrange_negative_data(anchor_labels, positive_labels, negative_labels, negative_data):
    samples, cols, opvals, joins = negative_data
    new_samples = []
    new_cols = []
    new_opvals = []
    new_joins = []
    new_negative_labels = []
    print(len(anchor_labels))
    print(len(positive_labels))
    print(len(negative_labels))
    print(len(negative_data))
    for i in range(len(anchor_labels)):
        difference = abs(anchor_labels[i] - positive_labels[i])
        large_difference = np.where(np.absolute(negative_labels - anchor_labels[i]) > difference)[0]
        select_idx = np.random.choice(large_difference, size=1)[0]
        new_samples.append(samples[select_idx])
        new_cols.append(cols[select_idx])
        new_opvals.append(opvals[select_idx])
        new_joins.append(joins[select_idx])
        new_negative_labels.append(negative_labels[select_idx])
    new_negative_data = [new_samples, new_cols, new_opvals, new_joins]
    return new_negative_data, new_negative_labels

def rearrange_negative_data_with_template(anchor_labels, positive_labels, negative_labels, anchor_template_labels, positive_template_labels, negative_template_labels, anchor_data, positive_data, negative_data, indices, large_difference):
    #anchor_sample_tensors, anchor_col_tensors, anchor_opval_tensors, anchor_join_tensors, anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks = anchor_data
    #positive_sample_tensors, positive_col_tensors, positive_opval_tensors, positive_join_tensors, positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks = positive_data
    #negative_sample_tensors, negative_col_tensors, negative_opval_tensors, negative_join_tensors, negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks = negative_data
    anchor_sample_tensors, anchor_predicate_tensors, anchor_join_tensors, anchor_sample_masks, anchor_predicate_masks, anchor_join_masks = anchor_data
    positive_sample_tensors, positive_predicate_tensors, positive_join_tensors, positive_sample_masks, positive_predicate_masks, positive_join_masks = positive_data
    negative_sample_tensors, negative_predicate_tensors, negative_join_tensors, negative_sample_masks, negative_predicate_masks, negative_join_masks = negative_data

    new_negative_samples = []
    #new_negative_cols = []
    #new_negative_opvals = []
    new_negative_predicates = []
    new_negative_joins = []
    new_negative_sample_masks = []
    #new_negative_col_masks =[]
    #new_negative_opval_masks = []
    new_negative_predicate_masks = []
    new_negative_join_masks = []
    new_negative_labels = []
    new_negative_template_labels = []
    print(len(anchor_labels))
    print(len(positive_labels))
    print(len(negative_labels))
    print(len(negative_data))
    for i in range(len(anchor_labels)):
        #difference = abs(anchor_labels[i] - positive_labels[i])
        #template_label = anchor_template_labels[i]
        #large_difference = np.where(np.absolute(negative_labels - anchor_labels[i]) > difference)[0]
        #same_templates = np.where(negative_template_labels == template_label)[0]
        #indices = list(set(large_difference).intersection(set(same_templates)))
        #for j in large_difference:
        #    if j in same_templates:
        #        indices.append(j)
        #if len(indices[i]) > 0:
        #    select_idx = np.random.choice(indices[i], size=1)[0]
        #else:
        select_idx = np.random.choice(large_difference[i], size=1)[0]
        new_negative_samples.append(negative_sample_tensors[select_idx])
        #new_negative_cols.append(negative_col_tensors[select_idx])
        #new_negative_opvals.append(negative_opval_tensors[select_idx])
        new_negative_predicates.append(negative_predicate_tensors[select_idx])
        new_negative_joins.append(negative_join_tensors[select_idx])
        new_negative_sample_masks.append(negative_sample_masks[select_idx])
        #new_negative_col_masks.append(negative_col_masks[select_idx])
        #new_negative_opval_masks.append(negative_opval_masks[select_idx])
        new_negative_predicate_masks.append(negative_predicate_masks[select_idx])
        new_negative_join_masks.append(negative_join_masks[select_idx])
        new_negative_labels.append(negative_labels[select_idx])
        new_negative_template_labels.append(negative_template_labels[select_idx])
    #new_negative_data = [new_samples, new_cols, new_opvals, new_joins]
    new_anchor_samples = torch.FloatTensor(anchor_sample_tensors)
    #new_anchor_cols = torch.FloatTensor(anchor_col_tensors)
    #new_anchor_opvals = torch.FloatTensor(anchor_opval_tensors)
    new_anchor_predicates = torch.FloatTensor(anchor_predicate_tensors)
    new_anchor_joins = torch.FloatTensor(anchor_join_tensors)
    new_anchor_sample_masks = torch.FloatTensor(anchor_sample_masks)
    #new_anchor_col_masks = torch.FloatTensor(anchor_col_masks)
    #new_anchor_opval_masks = torch.FloatTensor(anchor_opval_masks)
    new_anchor_predicate_masks = torch.FloatTensor(anchor_predicate_masks)
    new_anchor_join_masks = torch.FloatTensor(anchor_join_masks)
    new_anchor_labels = torch.FloatTensor(anchor_labels)
    new_anchor_template_labels = torch.FloatTensor(anchor_template_labels)

    new_positive_samples = torch.FloatTensor(positive_sample_tensors)
    #new_positive_cols = torch.FloatTensor(positive_col_tensors)
    #new_positive_opvals = torch.FloatTensor(positive_opval_tensors)
    new_positive_predicates = torch.FloatTensor(positive_predicate_tensors)
    new_positive_joins = torch.FloatTensor(positive_join_tensors)
    new_positive_sample_masks = torch.FloatTensor(positive_sample_masks)
    #new_positive_col_masks = torch.FloatTensor(positive_col_masks)
    #new_positive_opval_masks = torch.FloatTensor(positive_opval_masks)
    new_positive_predicate_masks = torch.FloatTensor(positive_predicate_masks)
    new_positive_join_masks = torch.FloatTensor(positive_join_masks)
    new_positive_labels = torch.FloatTensor(positive_labels)
    new_positive_template_labels = torch.FloatTensor(positive_template_labels)

    new_negative_samples = torch.FloatTensor(new_negative_samples)
    #new_negative_cols = torch.FloatTensor(new_negative_cols)
    #new_negative_opvals = torch.FloatTensor(new_negative_opvals)
    new_negative_predicates = torch.FloatTensor(new_negative_predicates)
    new_negative_joins = torch.FloatTensor(new_negative_joins)
    new_negative_sample_masks = torch.FloatTensor(new_negative_sample_masks)
    #new_negative_col_masks = torch.FloatTensor(new_negative_col_masks)
    #new_negative_opval_masks = torch.FloatTensor(new_negative_opval_masks)
    new_negative_predicate_masks = torch.FloatTensor(new_negative_predicate_masks)
    new_negative_join_masks = torch.FloatTensor(new_negative_join_masks)
    new_negative_labels = torch.FloatTensor(new_negative_labels)
    new_negative_template_labels = torch.FloatTensor(new_negative_template_labels)
    #current_dataset = dataset.TensorDataset(new_anchor_samples, new_anchor_cols, new_anchor_opvals, new_anchor_joins, \
    #                             new_anchor_sample_masks, new_anchor_col_masks, new_anchor_opval_masks, new_anchor_join_masks, new_anchor_labels, new_anchor_template_labels, \
    #                             new_positive_samples, new_positive_cols, new_positive_opvals, new_positive_joins, \
    #                             new_positive_sample_masks, new_positive_col_masks, new_positive_opval_masks, new_positive_join_masks, new_positive_labels, new_positive_template_labels, \
    #                             new_negative_samples, new_negative_cols, new_negative_opvals, new_negative_joins, \
    #                             new_negative_sample_masks, new_negative_col_masks, new_negative_opval_masks, new_negative_join_masks, new_negative_labels, new_negative_template_labels)    
    current_dataset = dataset.TensorDataset(new_anchor_samples, new_anchor_predicates, new_anchor_joins, \
                                 new_anchor_sample_masks, new_anchor_predicate_masks, new_anchor_join_masks, new_anchor_labels, new_anchor_template_labels, \
                                 new_positive_samples, new_positive_predicates, new_positive_joins, \
                                 new_positive_sample_masks, new_positive_predicate_masks, new_positive_join_masks, new_positive_labels, new_positive_template_labels, \
                                 new_negative_samples, new_negative_predicates, new_negative_joins, \
                                 new_negative_sample_masks, new_negative_predicate_masks, new_negative_join_masks, new_negative_labels, new_negative_template_labels)    
    return current_dataset, new_negative_labels
            
'''Currently Used'''
def get_generated_source_target_datasets(train_query_file, adaptation_query_file, train_bitmap_file, adaptation_bitmap_file, table2vec, join2vec, column2vec, op2vec, idxdicts, column_min_max_vals, min_val, max_val, \
                                         max_num_joins, max_num_predicates, templates, conn, cursor, model, num_samples=0, query_num=0, cuda='cpu'):
    #source_labels_train, source_labels_test, _, _, _, source_data_train, source_data_test, source_template_labels_train, source_template_labels_test, templates, template_distribution = encode_data_from_saver( \
    #    train_query_file, train_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)
    source_labels_train, source_labels_test, _, _, source_data_train, source_data_test, source_template_labels_train, source_template_labels_test, templates, template_distribution = encode_data_from_saver( \
        train_query_file, train_bitmap_file, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)

    #print(templates)
    #adaptation_labels, validation_labels, _, _, adaptation_data, validation_data, adaptation_template_labels, validation_template_labels, templates, template_distribution = encode_data_from_saver(adaptation_query_file, adaptation_bitmap_file, table2vec, join2vec, \
    #            column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)
    target_labels_train, target_labels_test, _, _, target_data_train, target_data_test, target_template_labels_train, target_template_labels_test, templates, template_distribution = encode_data_from_saver(adaptation_query_file, adaptation_bitmap_file, table2vec, join2vec, \
                column2vec, op2vec, column_min_max_vals, min_val, max_val, templates, num_samples=num_samples, query_num=query_num)
    
    source_samples_train, source_predicates_train, source_joins_train = source_data_train
    source_samples_test, source_predicates_test, source_joins_test = source_data_test
    target_samples_train, target_predicates_train, target_joins_train = target_data_train
    target_samples_test, target_predicates_test, target_joins_test = target_data_test

    adaptation_labels = []
    validation_labels = []
    adaptation_samples = []
    validation_samples = []
    adaptation_predicates = []
    validation_predicates = []
    adaptation_joins = []
    validation_joins = []
    adaptation_template_labels = []
    validation_template_labels = []

    for i in range(10):
        indices = np.random.choice(len(source_labels_train), size=512 - 50, replace=False)
        for i in range(50):
            adaptation_labels.append(target_labels_train[i])
            adaptation_samples.append(target_samples_train[i])
            adaptation_predicates.append(target_predicates_train[i])
            adaptation_joins.append(target_joins_train[i])
            adaptation_template_labels.append(target_template_labels_train[i])
        for i in indices:
            adaptation_labels.append(source_labels_train[i])
            adaptation_samples.append(source_samples_train[i])
            adaptation_predicates.append(source_predicates_train[i])
            adaptation_joins.append(target_joins_train[i])
            adaptation_template_labels.append(target_template_labels_train[i])
    '''
    for i in range(len(source_labels_train)):
        adaptation_labels.append(source_labels_train[i])
        #adaptation_labels.append(source_labels_train[int(i + len(source_labels_train) / 2) % len(source_labels_train)])
        adaptation_labels.append(target_labels_train[i])
        adaptation_samples.append(source_samples_train[i])
        #adaptation_samples.append(source_samples_train[int(i + len(source_labels_train) / 2) % len(source_labels_train)])
        adaptation_samples.append(target_samples_train[i])
        adaptation_predicates.append(source_predicates_train[i])
        #adaptation_predicates.append(source_predicates_train[int(i + len(source_labels_train) / 2) % len(source_labels_train)])
        adaptation_predicates.append(target_predicates_train[i])
        adaptation_joins.append(source_joins_train[i])
        #adaptation_joins.append(source_joins_train[int(i + len(source_labels_train) / 2) % len(source_labels_train)])
        adaptation_joins.append(target_joins_train[i])
        adaptation_template_labels.append(source_template_labels_train[i])
        #adaptation_template_labels.append(source_template_labels_train[int(i + len(source_labels_train) / 2) % len(source_labels_train)])
        adaptation_template_labels.append(target_template_labels_train[i])
    '''

    for i in range(len(source_labels_test)):
        validation_labels.append(source_labels_test[i])
        validation_labels.append(target_labels_test[i])
        validation_samples.append(source_samples_test[i])
        validation_samples.append(target_samples_test[i])
        validation_predicates.append(source_predicates_test[i])
        validation_predicates.append(target_predicates_test[i])
        validation_joins.append(target_joins_test[i])
        validation_joins.append(target_joins_test[i])
        validation_template_labels.append(source_template_labels_test[i])
        validation_template_labels.append(target_template_labels_test[i])

    adaptation_data = [adaptation_samples, adaptation_predicates, adaptation_joins]
    validation_data = [validation_samples, validation_predicates, validation_joins]

    #adaptation_dataset = make_single_dataset(adaptation_data, max_num_joins, max_num_cols, max_num_opvals, adaptation_labels)
    #adaptation_dataset = make_single_dataset(adaptation_data, max_num_joins, max_num_predicates, adaptation_labels)
    #adaptation_data_loader = DataLoader(adaptation_dataset, batch_size=1024)
    #adaptation_labels, _ =  predict_query(model, adaptation_data_loader, cuda, False)
    #adaptation_labels = tensor2numpy(adaptation_labels, cuda)
    #positive_data, positive_labels, positive_template_labels = generate_positive_queries(adaptation_data, idxdicts, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, adaptation_template_labels)
    #positive_dataset = make_single_dataset(positive_data, max_num_joins, max_num_predicates, positive_labels)
    #positive_data_loader = DataLoader(positive_dataset, batch_size=1024)
    #positive_labels, _ = predict_query(model, positive_data_loader, cuda, False)
    #positive_labels = tensor2numpy(positive_labels, cuda)
    positive_data = source_data_train
    positive_labels = source_labels_train
    positive_template_labels = source_template_labels_train
    #anchor_train_tensors = make_tensors(adaptation_data, max_num_joins, max_num_cols, max_num_opvals)
    #positive_train_tensors = make_tensors(positive_data, max_num_joins, max_num_cols, max_num_opvals)
    #negative_train_tensors = make_tensors(source_data_train, max_num_joins, max_num_cols, max_num_opvals)
    anchor_train_tensors = make_tensors(adaptation_data, max_num_joins, max_num_predicates)
    positive_train_tensors = make_tensors(positive_data, max_num_joins, max_num_predicates)
    negative_train_tensors = make_tensors(source_data_train, max_num_joins, max_num_predicates)

    positive_indices = []
    for i in range(len(adaptation_labels)):
        template_label = adaptation_template_labels[i]
        #print(positive_template_labels)
        #same_template_indices = np.where(np.array(positive_template_labels) == template_label)[0]
        #if len(same_template_indices) < 1:
        same_template_indices = np.arange(len(positive_template_labels))
        #print(same_template_indices)
        difference = np.argsort(np.absolute(positive_labels[same_template_indices] - adaptation_labels[i]))
        index = np.random.choice(difference[:int(0.3 * len(difference) + 1)], size=1)[0]
        positive_indices.append(same_template_indices[index])
    positive_samples = np.array(positive_data[0])[positive_indices]
    #positive_cols = np.array(positive_data[1])[positive_indices]
    #positive_opvals = np.array(positive_data[2])[positive_indices]
    positive_predicates = np.array(positive_data[1])[positive_indices]
    positive_joins = np.array(positive_data[2])[positive_indices]
    #positive_data = [positive_samples, positive_cols, positive_opvals, positive_joins]
    positive_data = [positive_samples, positive_predicates, positive_joins]
    positive_labels = np.array(positive_labels)[positive_indices]
    positive_template_labels = np.array(positive_template_labels)[positive_indices]
    #positive_train_tensors = make_tensors(positive_data, max_num_joins, max_num_cols, max_num_opvals)
    positive_train_tensors = make_tensors(positive_data, max_num_joins, max_num_predicates)
        

    train_indices = []
    train_large_difference = []
    for i in range(len(adaptation_labels)):
        #difference = abs(adaptation_labels[i] - positive_labels[i])
        difference = 0
        template_label = adaptation_template_labels[i]
        large_difference = np.where(np.absolute(source_labels_train - adaptation_labels[i]) > difference)[0]
        #same_templates = np.where(source_template_labels_train == template_label)[0]
        #indices = list(set(large_difference).intersection(set(same_templates)))
        indices = large_difference
        train_indices.append(indices)
        train_large_difference.append(large_difference)

    #negative_data, negative_labels = rearrange_negative_data_with_template(adaptation_labels, positive_labels, source_labels_train, adaptation_template_labels, positive_template_labels, source_template_labels_train, source_data_train)
    
    print("Finish training data")
    valid_positive_data, valid_positive_labels, valid_positive_template_labels = generate_positive_queries(validation_data, idxdicts, table2vec, join2vec, column2vec, op2vec, column_min_max_vals, validation_template_labels)
    #valid_positive_dataset = make_single_dataset(valid_positive_data, max_num_joins, max_num_cols, max_num_opvals, valid_positive_labels)
    valid_positive_dataset = make_single_dataset(valid_positive_data, max_num_joins, max_num_predicates, valid_positive_labels)
    valid_positive_data_loader = DataLoader(valid_positive_dataset, batch_size=1024)
    valid_positive_labels, _ = predict_query(model, valid_positive_data_loader, cuda, False)
    valid_positive_labels = tensor2numpy(valid_positive_labels, cuda)
    #anchor_test_tensors = make_tensors(validation_data, max_num_joins, max_num_cols, max_num_opvals)
    #positive_test_tensors = make_tensors(valid_positive_data, max_num_joins, max_num_cols, max_num_opvals)
    #negative_test_tensors = make_tensors(source_data_test, max_num_joins, max_num_cols, max_num_opvals)
    anchor_test_tensors = make_tensors(validation_data, max_num_joins, max_num_predicates)
    positive_test_tensors = make_tensors(valid_positive_data, max_num_joins, max_num_predicates)
    negative_test_tensors = make_tensors(source_data_test, max_num_joins, max_num_predicates)
    #valid_negative_data, valid_negative_labels = rearrange_negative_data_with_template(validation_labels, valid_positive_labels, source_labels_test, validation_template_labels, valid_positive_template_labels, source_template_labels_test, source_data_test)    
    print("Finish test data")

    test_indices = []
    test_large_difference = []
    for i in range(len(validation_labels)):
        #difference = abs(validation_labels[i] - valid_positive_labels[i])
        difference = 0
        template_label = validation_template_labels[i]
        large_difference = np.where(np.absolute(source_labels_test - validation_labels[i]) > difference)[0]
        #same_templates = np.where(source_template_labels_test == template_label)[0]
        #indices = list(set(large_difference).intersection(set(same_templates)))
        indices = large_difference
        test_indices.append(indices)
        test_large_difference.append(large_difference)
    #train_dataset = make_triplet_dataset(adaptation_data, adaptation_labels, adaptation_template_labels, positive_data, positive_labels, positive_template_labels, \
    #                                     source_data_train, source_labels_train, source_template_labels_train, max_num_joins, max_num_cols, max_num_opvals)
    #test_dataset = make_triplet_dataset(validation_data, validation_labels, validation_template_labels, valid_positive_data, valid_positive_labels, valid_positive_template_labels, \
    #                                    source_data_test, source_labels_test, source_template_labels_test, max_num_joins, max_num_cols, max_num_opvals)
    #return train_dataset, adaptation_labels, positive_labels, source_labels_train, adaptation_template_labels, positive_template_labels, source_template_labels_train, \
    #       test_dataset, validation_labels, valid_positive_labels, source_labels_test, validation_template_labels, valid_positive_template_labels, source_template_labels_test 
    return anchor_train_tensors, positive_train_tensors, negative_train_tensors, adaptation_labels, positive_labels, source_labels_train, adaptation_template_labels, positive_template_labels, source_template_labels_train, \
           anchor_test_tensors, positive_test_tensors, negative_test_tensors, validation_labels, valid_positive_labels, source_labels_test, validation_template_labels, valid_positive_template_labels, source_template_labels_test, train_indices, test_indices, train_large_difference, test_large_difference 

'''Currently used'''
def get_triplet_train_datasets(train_query_file, train_bitmap_file, adaptation_query_file, adaptation_bitmap_file, min_max_file, num_samples=0, repeat=10):
    #dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_data, \
    #            test_data, templates, template_distribution, template_labels_train, template_labels_test = templatize_and_encode_data(train_query_file, train_bitmap_file, min_max_file, num_samples=num_samples)
    dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, \
                test_data, templates, template_distribution, template_labels_train, template_labels_test = templatize_and_encode_data(train_query_file, train_bitmap_file,  adaptation_query_file, adaptation_bitmap_file, min_max_file, num_samples=num_samples, repeat=repeat)
    #train_tensors = make_tensors(train_data, max_num_joins, max_num_cols, max_num_opvals)
    #test_tensors = make_tensors(test_data, max_num_joins, max_num_cols, max_num_opvals)
    train_tensors = make_tensors(train_data, max_num_joins, max_num_predicates)
    test_tensors = make_tensors(test_data, max_num_joins, max_num_predicates)
    #return dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_tensors, test_tensors, templates, template_distribution, template_labels_train, template_labels_test
    return dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_tensors, test_tensors, templates, template_distribution, template_labels_train, template_labels_test

'''Currently used'''
def generate_epoch_data(train_tensors, test_tensors, train_labels, test_labels, train_template_labels, test_template_labels):
    datasets = []
    for data in [[train_tensors, train_labels, train_template_labels], [test_tensors, test_labels, test_template_labels]]:
        tensors, labels, template_labels = data
        #sample_tensors, col_tensors, opval_tensors, join_tensors, sample_masks, col_masks, opval_masks, join_masks = tensors
        sample_tensors, predicate_tensors, join_tensors, sample_masks, predicate_masks, join_masks = tensors
        order = np.argsort(labels)
        negative_order = np.zeros(len(order), dtype=int)
        positive_order = np.zeros(len(order), dtype=int)
        for i in range(len(order)):
            if i <= len(order) * 0.3:
                idx = random.randint(int(len(order) * 0.3) + i, len(order) - 1)
            elif i >= len(order) * 0.7:
                idx = random.randint(0, i - int(len(order) * 0.3))
            else:
                choice = random.randint(0, 1)
                if choice == 0:
                   idx = random.randint(0, i - int(len(order) * 0.3))
                else:
                   idx = random.randint(i + int(len(order) * 0.3), len(order) - 1)
            negative_order[order[i]] = order[idx]
        for i in range(len(order)):
            if i == 0:
                idx = random.randint(1, max(int(len(order) * 0.1), 1))
            elif i == len(order) - 1:
                idx = random.randint(int(len(order) * 0.9), len(order) - 1)
            else:
                choice = random.randint(0, 1)
                if choice == 0:
                    idx = random.randint(max(0, int(i - len(order) * 0.1)), i - 1)
                else:
                    idx = random.randint(i + 1, max(min(int(i + len(order) * 0.1), len(order) - 1), i + 1))
            idx = min(max(idx, 0), len(order) - 1)
            positive_order[order[i]] = order[idx]

        anchor_sample_tensors = torch.FloatTensor(sample_tensors)
        #anchor_col_tensors = torch.FloatTensor(col_tensors)
        #anchor_opval_tensors = torch.FloatTensor(opval_tensors)
        anchor_predicate_tensors = torch.FloatTensor(predicate_tensors)
        anchor_join_tensors = torch.FloatTensor(join_tensors)
        anchor_sample_masks = torch.FloatTensor(sample_masks)
        #anchor_col_masks = torch.FloatTensor(col_masks)
        #anchor_opval_masks = torch.FloatTensor(opval_masks)
        anchor_predicate_masks = torch.FloatTensor(predicate_masks)
        anchor_join_masks = torch.FloatTensor(join_masks)
        anchor_labels_tensor = torch.FloatTensor(labels)
        anchor_template_labels = torch.FloatTensor(template_labels)



        negative_sample_tensors = [sample_tensors[i] for i in negative_order]
        negative_sample_tensors = torch.FloatTensor(negative_sample_tensors)
        negative_sample_masks = [sample_masks[i] for i in negative_order]
        negative_sample_masks = torch.FloatTensor(negative_sample_masks)
        #negative_col_tensors = [col_tensors[i] for i in negative_order]
        #negative_col_tensors = torch.FloatTensor(negative_col_tensors)
        #negative_col_masks = [col_masks[i] for i in negative_order]
        #negative_col_masks = torch.FloatTensor(negative_col_masks)
        #negative_opval_tensors = [opval_tensors[i] for i in negative_order]
        #negative_opval_tensors = torch.FloatTensor(negative_opval_tensors)
        #negative_opval_masks = [opval_masks[i] for i in negative_order]
        #negative_opval_masks = torch.FloatTensor(negative_opval_masks)
        negative_predicate_tensors = [predicate_tensors[i] for i in negative_order]
        negative_predicate_tensors = torch.FloatTensor(negative_predicate_tensors)
        negative_predicate_masks = [predicate_masks[i] for i in negative_order]
        negative_predicate_masks = torch.FloatTensor(negative_predicate_masks)
        negative_join_tensors = [join_tensors[i] for i in negative_order]
        negative_join_tensors = torch.FloatTensor(negative_join_tensors)
        negative_join_masks = [join_masks[i] for i in negative_order]
        negative_join_masks = torch.FloatTensor(negative_join_masks)
        negative_labels = [labels[i] for i in negative_order]
        negative_labels_tensor = torch.FloatTensor(negative_labels)
        negative_template_labels = [template_labels[i] for i in negative_order]
        negative_template_labels = torch.FloatTensor(negative_template_labels)

        positive_sample_tensors = [sample_tensors[i] for i in positive_order]
        positive_sample_tensors = torch.FloatTensor(positive_sample_tensors)
        positive_sample_masks = [sample_masks[i] for i in positive_order]
        positive_sample_masks = torch.FloatTensor(positive_sample_masks)
        #positive_col_tensors = [col_tensors[i] for i in positive_order]
        #positive_col_tensors = torch.FloatTensor(positive_col_tensors)
        #positive_col_masks = [col_masks[i] for i in positive_order]
        #positive_col_masks = torch.FloatTensor(positive_col_masks)
        #positive_opval_tensors = [opval_tensors[i] for i in positive_order]
        #positive_opval_tensors = torch.FloatTensor(positive_opval_tensors)
        #positive_opval_masks = [opval_masks[i] for i in positive_order]
        #positive_opval_masks = torch.FloatTensor(positive_opval_masks)
        positive_predicate_tensors = [predicate_tensors[i] for i in positive_order]
        positive_predicate_tensors = torch.FloatTensor(positive_predicate_tensors)
        positive_predicate_masks = [predicate_masks[i] for i in positive_order]
        positive_predicate_masks = torch.FloatTensor(positive_predicate_masks)
        positive_join_tensors = [join_tensors[i] for i in positive_order]
        positive_join_tensors = torch.FloatTensor(positive_join_tensors)
        positive_join_masks = [join_masks[i] for i in positive_order]
        positive_join_masks = torch.FloatTensor(positive_join_masks)
        positive_labels = [labels[i] for i in positive_order]
        positive_labels_tensor = torch.FloatTensor(positive_labels)
        positive_template_labels = [template_labels[i] for i in positive_order]
        positive_template_labels = torch.FloatTensor(positive_template_labels)
        #positive_data_train = [positive_samples_train, positive_cols_train, positive_opvals_train, positive_joins_train]
    
        #current_dataset = dataset.TensorDataset(anchor_sample_tensors, anchor_col_tensors, anchor_opval_tensors, anchor_join_tensors, \
        #                         anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks, anchor_labels_tensor, anchor_template_labels, \
        #                         positive_sample_tensors, positive_col_tensors, positive_opval_tensors, positive_join_tensors, \
        #                         positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks, positive_labels_tensor, positive_template_labels, \
        #                         negative_sample_tensors, negative_col_tensors, negative_opval_tensors, negative_join_tensors, \
        #                         negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks, negative_labels_tensor, negative_template_labels)    
        current_dataset = dataset.TensorDataset(anchor_sample_tensors, anchor_predicate_tensors, anchor_join_tensors, \
                                 anchor_sample_masks, anchor_predicate_masks, anchor_join_masks, anchor_labels_tensor, anchor_template_labels, \
                                 positive_sample_tensors, positive_predicate_tensors, positive_join_tensors, \
                                 positive_sample_masks, positive_predicate_masks, positive_join_masks, positive_labels_tensor, positive_template_labels, \
                                 negative_sample_tensors, negative_predicate_tensors, negative_join_tensors, \
                                 negative_sample_masks, negative_predicate_masks, negative_join_masks, negative_labels_tensor, negative_template_labels)    
        datasets.append(current_dataset)
        print("Created TensorDataset for training data")
    train_dataset, test_dataset = datasets
    return train_dataset, test_dataset
