import torch

def spectral_loss (output, output2, target, tendency, wavenum_init, lamda_reg, time_step):

   loss1 = torch.mean((output-target)**2).cuda()
   
   # out_fft = torch.fft.rfft(output,dim=1).cuda()
   # target_fft = torch.fft.rfft(target,dim=1).cuda()
   
   # loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:])).cuda()


   out_du_fft =torch.fft.rfft((output-output2)/time_step,dim=1).cuda()
   target_du_fft =torch.fft.rfft(tendency/time_step,dim=1).cuda()

   loss3 = torch.mean(torch.abs(out_du_fft[:,0:]-target_du_fft[:,0:])).cuda()

   loss = loss1 + lamda_reg*loss3
  
   return loss