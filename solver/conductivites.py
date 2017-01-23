from fenics import *

class intracellular_conductivity2D(Expression): 
  def value_shape(self): 
    return (2,2)
  def eval(selv, value, x): 
    value[0] = 1.0 
    value[1] = 0.0 
    value[2] = 0.0 
    value[3] = 1.0 

class extracellular_conductivity2D(Expression): 
  def value_shape(self): 
    return (2,2)
  def eval(selv, value, x): 
    value[0] = 1.0 
    value[1] = 0.0 
    value[2] = 0.0 
    value[3] = 1.0 

class intracellular_conductivity3D(Expression): 
  def value_shape(self): 
    return (3,3)
  def eval(selv, value, x): 
    value[0] = 1.0 
    value[1] = 0.0 
    value[2] = 0.0 
    value[3] = 0.0 
    value[4] = 0.0 
    value[5] = 1.0 
    value[6] = 0.0 
    value[7] = 0.0 
    value[8] = 1.0 

class extracellular_conductivity3D(Expression): 
  def value_shape(self): 
    return (3,3)
  def eval(selv, value, x): 
    value[0] = 1.0 
    value[1] = 0.0 
    value[2] = 0.0 
    value[3] = 0.0 
    value[4] = 0.0 
    value[5] = 1.0 
    value[6] = 0.0 
    value[7] = 0.0 
    value[8] = 1.0 






def get_conductivities(nsd): 
  Mi = intracellular_conductivity2D() 
  Me = extracellular_conductivity2D()
  if nsd == 3: 
#    Mi = intracellular_conductivity3D() 
#    Me = extracellular_conductivity3D()
    Mi = Constant([[100,0,0], [0,100,0], [0,0,100]])
    Me = Constant([[100,0,0], [0,100,0], [0,0,100]])

  return Mi, Me  
