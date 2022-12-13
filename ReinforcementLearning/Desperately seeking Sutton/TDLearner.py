import numpy as np 			  		 			 	 	 		 		 	  		   	  			  	
import random as rand
import matplotlib.pyplot as plt 			  		 			 	 	 		 		 	  		   	  			  	

class TDLearner(object):
  def __init__(self, \
    random_walk_states=["A","B","C","D","E","F","G"],  # Number of random walk states A,B,C,D,E,F,G
    ideal_prediction = [1/6,1/3,1/2,2/3,5/6],  # Ideal prediction for right side termination default [1/6,1/3,1/2,2/3,5/6]
    ):
    self.random_walk_states = random_walk_states
    self.starting_walk_state = self.random_walk_states.index("D")
    self.terminate_states = [self.random_walk_states.index("A"),self.random_walk_states.index("G")]
    self.ideal_prediction = ideal_prediction
  def __next_state(self,current_state):
    walk_probability = np.random.uniform()
    return (current_state + (-1 * (walk_probability < 0.5) + \
                             1 * (walk_probability > 0.5) ))
  def __x_vector(self):
    X_v = np.zeros(len(self.random_walk_states))
    X_v[self.starting_walk_state] = 1
    walk_state = self.starting_walk_state
    walk = True
    while walk:
      X_t = np.zeros(len(self.random_walk_states))
      walk_state = self.__next_state(walk_state)
      X_t[walk_state] = 1
      X_v = np.vstack([X_v,X_t])
      walk = False if walk_state in self.terminate_states else True
    return X_v
  def __lambda_walk_matrix(self,walk_matrix,i,j):
    walk_matrix[i][j] = self.__x_vector()
  def walk_matrix(self,training_states,training_sequences):
    walk_matrix = np.array([ [None] * training_sequences] * training_states) 
    [ self.__lambda_walk_matrix(walk_matrix,i,j) for i in range(training_states) for j in range(training_sequences)]
    return walk_matrix
  def __lambda_DW(self,i,delta_weights,lambda_val, alpha,walk_matrix,Wt):
    try:
        delta_p = (Wt[int(np.argmax(walk_matrix[:-1,1:-1][i+1]))] - Wt[int(np.argmax(walk_matrix[:-1,1:-1][i]))])
    except IndexError:
        delta_p = (int(walk_matrix[-1][-1]) - Wt[int(np.argmax(walk_matrix[:-1,1:-1][i]))])
    t = i+1
    sum_gradient_Pt = np.sum(np.array([ np.power(lambda_val,(t-i)) for i in range(1,t+1)]).reshape(t,-1)  * np.array(walk_matrix[:,1:-1][:t]),axis=0 )
    delta_weights += alpha * delta_p  * sum_gradient_Pt
  def delta_weights(self,lambda_val, alpha,walk_matrix,Wt ):
    delta_weights = np.zeros(len(self.random_walk_states) - len(self.terminate_states))
    [ self.__lambda_DW(i,delta_weights,lambda_val, alpha,walk_matrix,Wt) for i in range(walk_matrix.shape[0]- 1)]
    return delta_weights
    
def testTDLearner():
  
  # Setup Random Walk Matrix..
  np.random.seed(18)
  TD = TDLearner()
  rwMatrix = TD.walk_matrix(100,10)
  
  # Run Experiment 1: Replication of Figure 3 from Sutton Paper
  print("Run Experiment 1: Replication of Figure 3 from Sutton Paper")
  lambda_vs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
  alpha=0.001
  Wt = np.array([0.5] * 5 )
  num_epochs=30
  training_sequences = 10 
  training_states = 100
  ideal_prediction = [1/6,1/3,1/2,2/3,5/6]
  rms_error = []
  epsilon = 1e-2
  for k,lambda_v in enumerate(lambda_vs):
    converged = False
    dv = []
    dv1 = 1e-2
    while not converged:
      for i in range(training_states):
          dw = np.zeros(5)
          for j in range(training_sequences):
              dw += TD.delta_weights(lambda_v, alpha,rwMatrix[i][j],Wt )
          Wt += dw
      dv.append(np.sum(Wt)**2)
      if len(dv)> 1: dv1 = np.absolute(dv[-1] - dv[-2])
      if dv1 < epsilon: converged = True
    rms_error.append(np.sqrt(np.sum(np.mean(np.power(ideal_prediction - Wt, 2)))))

  
  # Run Experiment 1: Replication of Figure 3 from Sutton Paper
  print("Plot Experiment 1: Replication of Figure 3 from Sutton Paper")
  plt.figure(figsize=(8,6))
  plt.plot(lambda_vs,rms_error,'-o')
  plt.xlabel('λ',size=15)
  plt.ylabel('Rms Error',size=15)
  plt.title('Sutton 1988- Figure 3',size=15)
  plt.savefig("Figure 3")
  plt.clf()
  
  # Run Experiment 2 A: Replication of Figure 4 from Sutton Paper
  print("Run Experiment 2 A: Replication of Figure 4 from Sutton Paper")
  lambda_vs = [1,0,0.8,0.3]
  alpha_vs = np.linspace(0,0.5,20)
  Wt = np.array([0.5] * 5 )
  rms_error = {key:[] for key in lambda_vs}
  for k,lambda_v in enumerate(lambda_vs):
    for l, alpha in enumerate(alpha_vs): 
      rms = 0
      for i in range(training_states):
          Wt = np.array([0.5] * 5 )
          for j in range(training_sequences):
              Wt += TD.delta_weights(lambda_v, alpha,rwMatrix[i][j],Wt )
          rms = np.sum(np.sqrt(np.sum(np.mean(np.power(ideal_prediction - Wt, 2)))))
      rms_error[lambda_v].append(rms)
    
  # Plot Experiment 2 A: Replication of Figure 4 from Sutton Paper
  print("Plot Experiment 2 A: Replication - Figure 4 from Sutton Paper")
  plt.figure(figsize=(10,6))
  [ plt.plot(alpha_vs,rms_error[i],'-o',label='λ = {}'.format(i)) for i in lambda_vs]
  plt.xlim([-0.05,0.8])
  plt.xticks(np.linspace(0,0.5,8))
  plt.xlabel('α',size=15)
  plt.ylabel('RMS Error',size=15)
  plt.title('Sutton 1988- Figure 4',size=15)
  plt.legend()
  plt.savefig("Figure 4")
  plt.clf()
    
  # Run Experiment 2 B: Replication of Figure 4 from Sutton Paper
  print("Run Experiment 2 B: Replication of Figure 4 from Sutton Paper")
  alpha_best = 0.4
  lambda_vs = np.linspace(0,1.0,20)
  rms_error = []
  for i in range(len(alpha_vs)):
    lambda_v = lambda_vs[i]
    rms = 0
    for i in range(training_states):
      Wt = np.array([0.5] * 5 )
      for j in range(training_sequences):
          Wt += TD.delta_weights(lambda_v, alpha_best,rwMatrix[i][j],Wt )
      rms = np.sum(np.sqrt(np.sum(np.mean(np.power(ideal_prediction - Wt, 2)))))
    rms_error.append(rms)
  
  # Plot Experiment 2 B: Replication of Figure 4 from Sutton Paper
  print("Plot Experiment 2 B: Replication - Figure 4 from Sutton Paper")
  plt.figure(figsize=(8,6))
  plt.plot(lambda_vs,rms_error,'-o')
  plt.xlabel('λ',size=15)
  plt.ylabel('RMS Error using best α',size=15)
  plt.title('Sutton 1988- Figure 5',size=15)
  plt.savefig("Figure 5")
  plt.clf()
  
  print("Done..")

if __name__=="__main__": 			  		 			 	 	 		 		 	  		   	  			  	
  print("Desparetely seeking Sutton")
  testTDLearner()  
