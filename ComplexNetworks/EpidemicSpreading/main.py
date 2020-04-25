from SIS import MontecarloSISModelSimulation
from Plotting import *
from ExperimentsDefinition import experiments


t_transitionary = 900
t_stationary = 100
mus = (.1, .25, .5, .75, .9)
p_0s = (.2, .05)
beta_increasing = .01
montecarlo_executions = 50

if __name__ == '__main__':
    for name, graph in experiments.items():
        # Create the folders where results will be saved
        path = os.path.join('Results', name)
        if not os.path.isdir(path):
            os.makedirs(path)
        # Save there the degree distributions of the graphs which will be used
        plot_degree_distribution(graph=graph,log_log=True, file='log-log', path=path, calculate_gamma='Scale Free' in name)
        plot_degree_distribution(graph=graph, file=None, path=path, calculate_gamma='Scale Free' in name)
        for p_0 in p_0s:
            for mu in mus:
                results = {}
                for beta in np.arange(0,1,beta_increasing):
                    results[np.round(beta,decimals=2)] = MontecarloSISModelSimulation(executions=montecarlo_executions, graph=graph, mu=mu,
                                                                                      beta=beta, p_0=p_0,
                                                                                      t=t_transitionary+t_stationary)
                    print("Beta: "+str(beta)+" finished")
                reduced_results = {key : value for key, value in results.items() if key in (.05, 0.1, 0.15, .2, .4, .8)}
                plot_evolution_over_time_by_beta(betas=reduced_results, mu=mu, p_0=p_0, model_definition = name,
                                                 path=path)
                plot_p_in_function_of_beta(ps=results, mu=mu, p_0=p_0, model_definition = name,
                                           t_stationary=t_stationary, path=path)



