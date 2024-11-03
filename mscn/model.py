import torch
import torch.nn as nn
import torch.nn.functional as F


# Define model architecture

class SetConv(nn.Module):
    def __init__(self, sample_feats=10, col_feats=10, opval_feats=10, join_feats=10, hid_units=256):
        super(SetConv, self).__init__()
        print("JOINS_FEATS: ", join_feats)
        print("COL_FEATS: ", col_feats)
        predicate_feats = col_feats + opval_feats
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        #self.col_mlp1 = nn.Linear(col_feats, hid_units)
        #self.col_mlp2 = nn.Linear(hid_units, hid_units)
        #self.opval_mlp1 = nn.Linear(opval_feats, hid_units)
        #self.opval_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        #self.part_mlp1 = nn.Linear(hid_units * 4, hid_units)
        self.part_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.part_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    #def forward(self, samples, cols, opvals, joins, sample_mask, col_mask, opval_mask, join_mask):
    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        '''
        hid_col = F.relu(self.col_mlp1(cols))
        hid_col = F.relu(self.col_mlp2(hid_col))
        hid_col = hid_col * col_mask
        hid_col = torch.sum(hid_col, dim=1, keepdim=False)
        col_norm = col_mask.sum(1, keepdim=False)
        hid_col = hid_col / col_norm

        hid_opval = F.relu(self.opval_mlp1(opvals))
        hid_opval = F.relu(self.opval_mlp2(hid_opval))
        hid_opval = hid_opval * opval_mask
        hid_opval = torch.sum(hid_opval, dim=1, keepdim=False)
        opval_norm = opval_mask.sum(1, keepdim=False)
        hid_opval = hid_opval / opval_norm
        '''

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        #hid = torch.cat((hid_sample, hid_col, hid_opval, hid_join), 1)
        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        feature = F.relu(self.part_mlp1(hid))
        #hid_opval = F.relu(self.part_mlp2(hid_opval))
        #hid = torch.cat((feature, hid_opval), 1)
        hid = F.relu(self.out_mlp1(feature))
        pred = torch.sigmoid(self.out_mlp2(hid))
        return pred, feature


class LinearModel(nn.Module):
    def __init__(self, sample_feats=10, col_feats=10, join_feats=10, hid_units=256):
        super(LinearModel, self).__init__()
        print("JOINS_FEATS: ", join_feats)
        print("COL_FEATS: ", col_feats)
        self.rep_mlp1 = nn.Linear(sample_feats + col_feats * 2 + join_feats, hid_units)
        self.rep_mlp2 = nn.Linear(hid_units, hid_units)
        #self.col_mlp1 = nn.Linear(col_feats, hid_units)
        #self.col_mlp2 = nn.Linear(hid_units, hid_units)
        #self.opval_mlp1 = nn.Linear(opval_feats, hid_units)
        #self.opval_mlp2 = nn.Linear(hid_units, hid_units)
        self.reg_mlp1 = nn.Linear(hid_units, hid_units)
        self.reg_mlp2 = nn.Linear(hid_units, 1)

    #def forward(self, samples, cols, opvals, joins, sample_mask, col_mask, opval_mask, join_mask):
    def forward(self, X):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_rep = F.relu(self.rep_mlp1(X))
        feature = F.relu(self.rep_mlp2(hid_rep))
        hid_reg = F.relu(self.reg_mlp1(feature))
        pred = torch.sigmoid(self.reg_mlp2(hid_reg))
        return pred, feature
