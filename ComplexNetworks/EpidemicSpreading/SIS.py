import numpy as np
import networkx as nx
from multiprocessing.pool import ThreadPool

NODE, INFECTION = 0, 1

def SISModelSimulation(neighbors, mu, beta, p_0, t):
    """
    Every node is an individual which can be in either state S or state I,
    the time is discrete, and at each time step each node contacts (synchronously) with all of its neighbors.
    :param neighbors: Adjacency list of the nodes
    :param mu: Spontaneous recovery probability.
    :param beta: Infection probability of a susceptible (S) individual when it is contacted by an infected (I) one
    :param p_0: Initial percentage of nodes infected.
    :param t: Time steps which the model will simulate

    :return:
        p: Evolution of p, the mean of nodes infected at the stationary state
    """
    # Infected if obtains a random value lower than beta between N tries. Where N is the amount of infected neighbors
    infection_condition = lambda node, infecteds :\
                                 np.any(np.random.uniform(size=np.sum(np.isin(neighbors[node], infecteds))) <= beta)

    # ------------------------ Virus Appears ------------------------
    #Virus appears, infecting suddenly a p_0 fraction of the population
    state = np.array([range(len(neighbors)), np.random.uniform(size=len(neighbors))<=p_0], dtype=np.int16)
    p = [p_0]

    # -------------------------- Time Runs --------------------------
    for t in range(t):
        #Separate the population among Susceptibles (S) and Infecteds (I) subjects
        susceptibles = state[NODE,state[INFECTION] == False]
        infecteds = state[NODE, state[INFECTION] == True]

        # ------------------ Healing Step ------------------
        # The ones which being Infected (I) have enough luck will be healed at next time step
        next_survivors = infecteds[np.random.uniform(size=len(infecteds))<=mu]

        # ----------------- Spreading Step -------------------
        # The ones which being Susceptible (S) have evil companions (I) and bad luck will be infected at next time step
        next_infecteds = [susceptible for susceptible in susceptibles
                         if infection_condition(node=susceptible,infecteds=infecteds)]

        # ------------------ Update Step -------------------
        state[INFECTION, next_survivors] = False
        state[INFECTION, next_infecteds] = True

        # Save results
        p.append(np.mean(state[INFECTION]))

    return p

def MontecarloSISModelSimulation(executions, graph, mu, beta, p_0, t, threads=8):
    """
    Executes the SIS modelSimulation in a Montecarlo way
    :param executions: Amount of executions to repeat
    :param graph: graph as nx object
    :param mu: Spontaneous recovery probability.
    :param beta: Infection probability of a susceptible (S) individual when it is contacted by an infected (I) one
    :param p_0: Initial percentage of nodes infected.
    :param t: Time steps which the model will simulate

    :return:
        p: Mean Evolution of p over executions iterations
    """
    # -------------------- Efficiency Preparation -------------------
    # In order to avoid slow non numpy computations during the bucles graph information is extracted
    # Adapts the graph for ensuring consistent translation to ordered indexes
    graph = nx.OrderedGraph(nx.convert_node_labels_to_integers(G=graph))
    # Generates the complete graph adjacency list
    neighbors = [np.array(list(graph.neighbors(node))) for node in graph.nodes()]

    # ------------- Multithreading Montecarlo Execution -------------
    ps = []
    with ThreadPool(processes=None) as pool:  # use all cpu cores
        for i in range(executions//threads):
            async_results = [pool.apply_async(SISModelSimulation, (neighbors,mu, beta, p_0, t)) for _ in range(threads)]
            ps.append(np.mean(np.array([async_result.get() for async_result in async_results], dtype=np.float32),axis=0))
            if i%2==0:
                print("Execution "+str(i*threads)+"/"+str(executions))
                
        pool.close()
        pool.terminate()
        pool.join()

    return np.mean(ps, axis=0)