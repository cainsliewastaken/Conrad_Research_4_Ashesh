import scipy.io


path_outputs = '/media/volume/sdb/conrad_stability/RMSE_data/'
eval_output_name = 'MLP_predicted_'+'PECstep'+'_tendnecy_RMSE'


mat_file = {}
scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_tendency/predicted_PECstep_1024_tendency_lead1.mat', mat_file)
data = {}
data[u'RMSE'] = mat_file['RMSE']
scipy.io.savemat(path_outputs+eval_output_name+'.mat', data)
