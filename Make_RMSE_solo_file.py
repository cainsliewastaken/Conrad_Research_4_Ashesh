import scipy.io


path_outputs = '/media/volume/sdb/conrad_stability/RMSE_data/'
eval_output_name = 'FNO_predicted_'+'Directstep'+'_RMSE'


mat_file = {}
scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_Directstep_1024_FNO_lead1.mat', mat_file)
data = {}
data[u'RMSE'] = mat_file['RMSE']
scipy.io.savemat(path_outputs+eval_output_name+'.mat', data)
