"""Freeze weights; intervene only on the complete transport projection output."""
import hashlib
import torch

def state_digest(model):
 h=hashlib.sha256()
 for key,t in sorted(model.state_dict().items()):
  t=t.detach().cpu().contiguous()
  h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
  h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def traced_forward(model,batch,disabled=False):
 trace={'content':[],'transport':[],'anchor':[]};handles=[]
 def capture(name,zero=False):
  def hook(module,args,output):
   trace[name].append(output.detach().clone())
   return torch.zeros_like(output) if zero else None
  return hook
 try:
  handles.append(model.frame_values.register_forward_hook(capture('content')))
  handles.append(model.transport_values.register_forward_hook(capture('transport',disabled)))
  handles.append(model.reference.register_forward_hook(capture('anchor')))
  with torch.inference_mode():pred=model(batch)
 finally:
  for handle in handles:handle.remove()
 assert len(trace['content'])==len(trace['transport'])==2 and len(trace['anchor'])==1
 return pred,trace

def both(model,batch):
 before,t=traced_forward(model,batch);off,u=traced_forward(model,batch,True)
 assert all(torch.equal(a,b) for k in t for a,b in zip(t[k],u[k]))
 hist=(batch['mask']&(batch['type']==1)).any(1)
 expected=(torch.stack(t['transport'],1)@model.output_basis.T)*hist[:,None,None]
 delta=float(torch.max(torch.abs((before-off)-expected)))
 assert delta<3e-5
 if (~hist).any():assert torch.equal(before[~hist],off[~hist])
 return {'before':before.numpy(),'transport_off':off.numpy()},delta
