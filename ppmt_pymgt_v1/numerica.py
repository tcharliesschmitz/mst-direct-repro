"""Caudas lineares robustas; aritmética interior idêntica ao pymgt original.
O ajuste, a busca das direções e os estados são produzidos pela biblioteca.
"""
import numpy as np
from pymgt.utils import orthonormal_basis

def interpolate(x,xp,fp):
 xp=np.asarray(xp,dtype=float);fp=np.asarray(fp,dtype=float);x=np.asarray(x,dtype=float)
 if np.any(np.diff(xp)<=0) or np.any(np.diff(fp)<=0):raise ValueError('Tabela não estritamente monótona; não agregar silenciosamente.')
 y=np.interp(x,xp,fp);k=min(len(xp)-1,max(8,int(.01*len(xp))))
 lo=x<xp[0];hi=x>xp[-1]
 y[lo]=fp[0]+(x[lo]-xp[0])*(fp[k]-fp[0])/(xp[k]-xp[0])
 y[hi]=fp[-1]+(x[hi]-xp[-1])*(fp[-1]-fp[-k-1])/(xp[-1]-xp[-k-1])
 return y

class StableTables:
 def __init__(self,model):
  self.state=model.state;self.rotations=[(orthonormal_basis(s.direction),s.raw_table) for s in self.state.iteration_steps];self.invrotations=[np.linalg.inv(q) for q,_ in self.rotations];self.S=self.state.state_sphering.sphering_matrix;self.Sinv=np.linalg.inv(self.S);self.mu=self.state.state_sphering.means
 def marginal(self,x,state,inverse=False):
  if inverse:return np.column_stack([interpolate(x[:,j],state.gaussian_table,state.raw_tables[:,j]) for j in range(x.shape[1])])
  return np.column_stack([interpolate(x[:,j],state.raw_tables[:,j],state.gaussian_table) for j in range(x.shape[1])])
 def forward(self,x):
  y=self.marginal(np.asarray(x),self.state.state_initial_marginal);y=(y-self.mu)@self.S
  for (q,table),qi in zip(self.rotations,self.invrotations):
   z=y@q;z[:,0]=interpolate(z[:,0],table,self.state.gaussian_table);y=z@qi
  return self.marginal(y,self.state.state_final_marginal)
 def inverse(self,y):
  x=self.marginal(np.asarray(y),self.state.state_final_marginal,True)
  for (q,table),qi in zip(reversed(self.rotations),reversed(self.invrotations)):
   z=x@q;z[:,0]=interpolate(z[:,0],self.state.gaussian_table,table);x=z@qi
  return self.marginal(x@self.Sinv+self.mu,self.state.state_initial_marginal,True)
