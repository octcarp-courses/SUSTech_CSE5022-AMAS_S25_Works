package camera.environment

import groovy.transform.CompileStatic

@CompileStatic
class VisionGraph {
    Map<Integer, Map<Integer, Double>> pheromoneMap = [:].withDefault { [:].withDefault { 0.0
        } }
    final double rho = 0.1  // evaporation rate
    final double delta = 1.0  // pheromone increase per trade

    void evaporate() {
        pheromoneMap.each { from, neighbors ->
            neighbors.each { to, value ->
                pheromoneMap[from][to] = value * (1 - rho)
            }
        }
    }

    void reinforce(int fromId, int toId) {
        pheromoneMap[fromId][toId] = pheromoneMap[fromId][toId] * (1 - rho) + delta
    }

    List<Integer> getNeighbors(int cameraId) {
        pheromoneMap[cameraId].findAll { it.value > 0.01 }.keySet().toList()
    }

    double getNotifyProbability(int fromId, int toId) {
        def total = pheromoneMap[fromId].values().sum() as double
        def pheromoneValue = pheromoneMap[fromId][toId]

        double result
        if (total > 0) {
            result = pheromoneValue / total
        } else {
            result = 0.0
        }

        return result
    }

    List<Integer> getActiveNeighbors(int fromId, double threshold = 0.01) {
        Map<Integer, Double> neighbors = pheromoneMap[fromId]
        List<Integer> active = []
        neighbors.each { Integer toId, Double value ->
            if (value >= threshold) {
                active << toId
            }
        }
        return active
    }

    void updatePheromone(int fromId, int toId, boolean tradeOccurred, double rho, double delta) {
        Map<Integer, Double> neighbors = pheromoneMap[fromId]

        double currentLevel = 0.0
        if (neighbors.containsKey(toId)) {
            currentLevel = neighbors[toId]
        }

        double newLevel = tradeOccurred ?
                (1 - rho) * currentLevel + delta :
                (1 - rho) * currentLevel

        neighbors[toId] = newLevel
        pheromoneMap[fromId] = neighbors
    }
}
