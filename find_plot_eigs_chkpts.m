Direct_struct = load('MLP_KS_Directstep_lead1_jacs_all_chkpts.mat');
PEC_struct = load('MLP_KS_PECstep_lead1_jacs_all_chkpts.mat');

direct_mat = cell2mat(struct2cell(Direct_struct));
PEC_mat = cell2mat(struct2cell(PEC_struct));