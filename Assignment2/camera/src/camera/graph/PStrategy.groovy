package camera.graph

import groovy.transform.CompileStatic

/**
 * Notify probability strategy enumerate
 * 
 * Two approach: Smooth or step
 */
@CompileStatic
enum PStrategy {
    SMOOTH,
    STEP,
}
