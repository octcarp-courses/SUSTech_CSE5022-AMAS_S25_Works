package camera.common

import groovy.transform.CompileStatic

/**
 * Bid record POJO
 */
@CompileStatic
class BidRec {
    int bidderId
    int auctioneerId
    int targetId
    double bid

    BidRec(int bidderId, int auctioneerId, int targetId, double bid) {
        this.bidderId = bidderId
        this.auctioneerId = auctioneerId
        this.targetId = targetId
        this.bid = bid
    }

    @Override
    String toString() {
        return "Bid ${bid} for target ${targetId}, from bidder ${bidderId} to ${auctioneerId}";
    }
}
