import numpy as np
import psycopg2

def idx_to_onehot(idx, number):
    onehot = np.zeros(number, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot

def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing

def get_all_names(source):
    all_names = []
    for query in source:
        for it in query:
            if it not in all_names:
                all_names.append(it)
    return all_names

def get_all_names_predicates(predicates):
    columns = []
    operators = []
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                col = predicate[0]
                op = predicate[1]
                if col not in columns:
                    columns.append(col)
                if op not in operators:
                    operators.append(op)
    return columns, operators

def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    if val_norm > 1 or val_norm < 0:
        print("ERROR {} {} {} {}".format(column_name, val_norm, max_val, min_val))
    return np.array(val_norm, dtype=np.float32)

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val, cuda):
    if cuda == 'cuda':
        labels_norm = [l.cpu() for l in labels_norm]
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)

def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            if len(samples[i]) > 0:
                sample_vec.append(samples[i][j])
            else:
                sample_vec.append([])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc

def query_encode_linear(samples, tables, predicates, joins, column_min_max_vals, table2vec, column2vec, op2vec, join2vec, num_materialized_samples):
    encodes = []
    if num_materialized_samples > 0:
        sample_size = len(samples[0][0])
    for i, query in enumerate(predicates):
        enc = [0 for _ in range(len(join2vec))]
        for _ in range(len(column2vec)):
            enc.append(0)
            enc.append(1)
        for predicate in query:
            if len(predicate) == 3:
                column = predicate[0]
                op = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)
                if norm_val > 1:
                    print("ERROR")
                cid = column2vec[column]
                if op is '=':
                    enc[len(join2vec) + 2 * cid] = norm_val
                    enc[len(join2vec) + 2 * cid + 1] = norm_val
                elif op is '<' or op is '<=':
                    enc[len(join2vec) + 2 * cid + 1] = norm_val
                else:
                    enc[len(join2vec) + 2 * cid] = norm_val
        for predicate in joins[i]:
            jid = join2vec[predicate]
            enc[jid] = 1
        if num_materialized_samples > 0:
            for t in table2vec:
                if t in tables:
                    idx = tables.idx(t)
                    enc = enc + samples [i][idx]
                else:
                    enc = enc + [0] * sample_size
        encodes.append(enc)
    return encodes

def encode_predicate_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicate_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        predicate_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros(len(column2vec) + len(op2vec) + 1)

            predicate_enc[i].append(pred_vec)
        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicate_enc, joins_enc


def encode_col_opval_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    col_enc = []
    opval_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        col_enc.append(list())
        opval_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                col_vec = []
                opval_vec = []
                col_vec.append(column2vec[column])
                opval_vec.append(op2vec[operator])
                opval_vec.append(norm_val)
                opval_vec = np.hstack(opval_vec)
            else:
                col_vec = np.zeros(len(column2vec))
                opval_vec = np.zeros((len(op2vec) + 1))

            col_enc[i].append(col_vec)
            opval_enc[i].append(opval_vec)

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return col_enc, opval_enc, joins_enc

def pg_initialize(database, tables):
    conn = psycopg2.connect(database=database, port='5433', user='peizhi')
    conn.autocommit = True
    cursor = conn.cursor()

    for t in tables:
        cursor.execute('analyze ' + t + ';')
        conn.commit()

    return conn, cursor
    
