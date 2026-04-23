# ADMET PXR Challenge

# Activity Prediction
## models:
- message-passing graph neural network (MP-GNN) with multi-layer perceptron (MLP) regression head.
    - node feats: atomic #, degree, formal charge, hybridization, aromatic_boolean, # of H's
    - edge feats: single_bond_boolean, double_bond_boolean, triple_bond_boolean, aromatic_bond_boolean, is_conjugated_boolea, is_in_ring_boolean
    - graph feats: mol_wt, logp, num_h_donors, num_h_acceptors, TPSA, num_rotatable_bonds, Emax

