package tileworld.utils

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.space.graph.Network
import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import repast.simphony.util.ContextUtils
import tileworld.common.GridTrait
import tileworld.environment.Tile

@CompileStatic
class GridUtils {
    private static GridUtils instance = null

    private GridUtils() {}

    static GridUtils getInstance() {
        if (!instance) {
            synchronized (GridUtils) {
                if (!instance) instance = new GridUtils()
            }
        }
        instance
    }

    static int calcPath(GridPoint a, GridPoint b) {
        Math.abs(a.x - b.x) + Math.abs(a.y - b.y)
    }

    static int calcPath(GridTrait a, GridTrait b) {
        calcPath(a.location, b.location)
    }

    def <T extends GridTrait> List<T> getObjectsWithRadius(GridTrait source, Class<T> targetClz, int radius) {
        Context context = ContextUtils.getContext(source)
        Grid grid = source.grid
        int sourceX = source.location.x
        int sourceY = source.location.y

        context.getObjects(targetClz)
                .findAll { obj -> 
                    def gridObj = obj as GridTrait
                    GridPoint objLoc = gridObj.location
                    (Math.abs(objLoc.x - sourceX) <= radius) && (Math.abs(objLoc.y - sourceY) <= radius)
                }
    }
}

