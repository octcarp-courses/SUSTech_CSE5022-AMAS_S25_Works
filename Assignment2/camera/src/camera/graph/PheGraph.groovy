package camera.graph

import camera.utils.ParameterUtils
import groovy.transform.CompileStatic

/**
 * Pheromone vision graph
 */
@CompileStatic
class PheGraph {
    private final int dim

    // Map for pheromone
    private final Map<Integer, Map<Integer, Double>> pheromoneMap = [:].withDefault {
        [:].withDefault {
            0.5
        }
    }

    // Map for last trade infomation
    private final Map<Integer, Map<Integer, Boolean>> tradeMap = [:].withDefault {
        [:].withDefault {
            false
        }
    }

    // pheromone evaporation rate
    final double RHO = ParameterUtils.instance.PHEROMONE_RHO
    // pheromone increase per trade
    final double DELTA = ParameterUtils.instance.PHEROMONE_DELTA

    // probability threshold to notify others
    final double EPS = ParameterUtils.instance.PROBABILITY_EPS
    // probability threshold to notify weak neighbors
    final double ETA = ParameterUtils.instance.PROBABILITY_ETA

    final PStrategy pStrategy

    PheGraph(int dim) {
        pStrategy = PStrategy.STEP
        this.dim = dim
        initGraph()
    }

    PheGraph (PStrategy pStrategy){
        this.pStrategy = pStrategy
    }


    Map<Integer, Map<Integer, Double>> graphSnapshot(){
        Map<Integer, Map<Integer, Double>> snapshot = [:]
        pheromoneMap.each { outerKey, outerValue ->
            snapshot[outerKey] = [:]
            outerValue.each { innerKey, innerValue ->
                snapshot[outerKey][innerKey] = innerValue
            }
        }
        snapshot
    }

    /**
     * Initial for new step
     */
    void initThisStep() {
        // Clear the trade record map
        (1..dim).each { i ->
            tradeMap.put(i, [:])
            (1..dim).each{ j ->
                if (i != j) {
                    tradeMap[i][j] = false
                }
            }
        }
    }
    
    /**
     * Initialize graph when be created
     */
    void initGraph() {
        (1..dim).each { i ->
            pheromoneMap.put(i, [:])
            (1..dim).each{ j ->
                if (i != j) {
                    pheromoneMap[i].put(j, 0.5d)
                }
            }
        }

        (1..dim).each { i ->
            tradeMap.put(i, [:])
            (1..dim).each{ j ->
                if (i != j) {
                    tradeMap[i].put(j, false)
                }
            }
        }
    }

    /**
     * Evaporate based on last step infomation
     */
    void evaporateLastStep() {
        // For each element
        pheromoneMap.each { from, neighbors ->
            neighbors.each { to, value ->
                // Last time have trade?
                boolean tradeOccurred = tradeMap[from][to]
                // Determine the pheromone
                double newLevel = tradeOccurred ?
                        (1 - RHO) * value + DELTA :
                        (1 - RHO) * value
                // Update the value
                neighbors[to] = newLevel
            }
        }
    }

    /**
     * Last time trade record
     * 
     * @param fromId from auctioneer
     * @param toId to bidder winner
     */
    void reinforce(int fromId, int toId) {
        // Record the trade info
        tradeMap[fromId][toId] = true
    }

    /**
     * Method to get neighbor's notify probability for specific camera
     * 
     * @param fromId From which camera
     * @return A Integer -> Double map, indicate neighbor's id —> notify probability
     */
    Map<Integer, Double> getNotifyProbabilities(int fromId) {
        // Get all it's neighbors as a list
        def neighbors = pheromoneMap[fromId]

        // Initialize the result map
        Map<Integer, Double> probabilities = [:].withDefault{ 0.0 }

        // Select strategy
        switch(pStrategy) {
            case PStrategy.SMOOTH:
            // SMOOTH strategy
            // Max neighbors' pheromone value im
                double im = pheromoneMap[fromId].values().max() as double

                if (im == 0) {
                    // No available neighbor, broadcast
                    neighbors.each{ id, phe ->
                        // All probabilities are 1
                        probabilities[id] = 1d
                    }
                } else {
                    neighbors.each{ id, phe ->
                        // Use Eq. 4 from paper
                        probabilities[id] = (1d + phe) / (1d + im)
                    }
                }
                break
            case PStrategy.STEP:
            // STEP strategy
                neighbors.each{ id, phe ->
                    // Use Eq. 5 from paper
                    probabilities[id] = (phe > EPS ? 1d : ETA)
                }
        }

        return probabilities
    }
}
