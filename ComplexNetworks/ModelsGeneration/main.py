import numpy as np
import os

from Analyzer import Analyzer
from ExperimentsDefinition import getAllExperiments, poisson_dist, ws_dist, power_law_dist

# Constants
results_path = 'results'

if __name__ == '__main__':

    # Iterate over all the experiments defined in 'ExperimentsDefinition'
    for exp_name, function, parameters, layout in getAllExperiments():
        # Initialize the graph as the network obtained by the current model
        model = Analyzer(graph=function(**parameters))

        # Define the folder name of the current experiment by its parameters
        if "P_k" not in parameters:
            printable_parameters = str(parameters)[1:-1].replace(":", "=")
        else:  # If P_k appears, set it followed by a random number
            printable_parameters = 'N=' + str(parameters['N']) + 'P_k-SetIt-' + str(np.random.randint(0, 100))

        # Set the full path of the experiment
        path = os.path.join(results_path, exp_name, printable_parameters)

        # Create all the folders of the path that do not exist yet
        if not os.path.isdir(path):
            os.makedirs(path)

        # Plot the obtained PDF and CCDF histograms of the network in log-log and linear scale
        model.plot_distribution(log_log=True, file='Experimental (log-log)', path=path)
        model.plot_distribution(file='Experimental', path=path)

        # Plot the theoretical PDF and CCFF histograms of the tested model according to its parameters
        if exp_name == 'Erdós-Rényi N-K':
            model.plot_distribution(file='Theoretical', path=path,
                                    plot_histogram=poisson_dist(parameters['N'], 2*parameters['K']/parameters['N']))

        elif exp_name == 'Erdós-Rényi N-p':
            model.plot_distribution(file='Theoretical', path=path,
                                    plot_histogram=poisson_dist(parameters['N'], parameters['p']*parameters['N']))

        elif exp_name == 'Watts-Strogatz':
            if not np.isclose(parameters['p'], .0):  # Regular case: Watts-Strogatz distribution
                model.plot_distribution(file='Theoretical', path=path,
                                        plot_histogram=ws_dist(parameters['N'], parameters['k'], parameters['p']))
            else:  # Special case: Dirac delta function
                model.plot_distribution(file='Theoretical', path=path,
                                        plot_histogram=np.array([[0, 0], [parameters['k'], 1], [2*parameters['k'], 0]]))

        elif exp_name == 'Barabási-Albert':
            model.plot_distribution(log_log=True, file='Theoretical (log-log)', path=path,
                                    plot_histogram=power_law_dist(parameters['N'], 3))

        elif exp_name == 'Configuration Model':
            model.plot_distribution(log_log=True, file='Theoretical (log-log)', path=path,
                                    plot_histogram=parameters['P_k'])
            model.plot_distribution(file='Theoretical', path=path, plot_histogram=parameters['P_k'])

        # Plot the graph of the network
        #model.plot_graph(layout=layout, alg_name=exp_name, parameters=parameters, save_at=path)

        model.save_graph(path=path)  # Save the network
