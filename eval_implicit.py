
M=20000
pred = np.zeros([M,np.size(label_test,1)])
for k in range(0,M):
 
    if (k==0):

        out =(PEC4step(mynet,input_test_torch[0,:]))
        out = implicit_iterations(mynet,input_test_torch[0,:].cuda(),out,50)
        pred [k,:] = out.detach().cpu().numpy()

    else:

        out = (PEC4step(mynet,torch.from_numpy(pred[k-1,:]).float().cuda()))
        out = implicit_iterations(mynet,torch.from_numpy(pred[k-1,:]).float().cuda(),out,50)
        pred [k,:] = out.detach().cpu().numpy()

matfiledata = {}
matfiledata[u'prediction'] = pred
matfiledata[u'Truth'] = label_test 
hdf5storage.write(matfiledata, '.', path_outputs+'predicted_KS_Spectral_Loss_PEC4step_with_implicit_step_'+'lambda_reg_'+str(lamda_reg)+'_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Predictions')