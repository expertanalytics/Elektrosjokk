from fenics import *

class shock2D(Expression): 
  def __init__(self,t): 
    self.t = t 
  def eval(self, value, x): 
    # print "in eval ", x[0], x[1], self.t 
    if self.t <= 0.1: 
      value[0] = exp(-100*pow(x[0]-0.5, 2))*exp(-100*pow(x[1]-1, 2)) \
               - exp(-100*pow(x[0]-0.0, 2))*exp(-100*pow(x[1]-0.5, 2))    
    else: 
      value[0] = 0     


class shock3D(Expression): 
  def __init__(self,t): 
    self.t = t 
  def eval(self, value, x): 
#    print "in eval ", x[0], x[1], self.t 
    if self.t <= 0.1: 
      value[0] = exp(-100*pow(x[0]-34.6, 2))*exp(-100*pow(x[1]-0.28, 2))*exp(-100*pow(x[2]-87.0)) \
                  
    else: 
      value[0] = 0     



def get_shock(t=0): 
  S = Expression(
      # "700*exp(-0.003*pow(x[0]-34.6, 2))*exp(-0.003*pow(x[1]-0.28, 2))*exp(-0.003*pow(x[2]-87.0, 2))*exp(-t)",
      "700*exp(-0.003*pow(x[0]-10, 2))*exp(-0.003*pow(x[1]-68, 2))*exp(-0.003*pow(x[2]-32.0, 2))*exp(-t)",
      t=t,
      degree=1)
  return S  
