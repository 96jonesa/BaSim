/**
 * Represents the position of something in a {@link BarbarianAssaultMap}.
 */
export class Position {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    /**
     * Determines if this position has the same coordinates as the given position.
     *
     * @param position  the position to determine if it has the same coordinates as this position
     * @return          true if this position has the same coordinates as the given position,
     *                  otherwise false
     */
    equals(position) {
        return this.x === position.x && this.y === position.y;
    }
    /**
     * Gets the (L-infinity) distance from this position to the given position.
     *
     * @param position  the position to get the (L-infinity) distance of from this position
     */
    distance(position) {
        return Math.max(Math.abs(this.x - position.x), Math.abs(this.y - position.y));
    }
    euclideanDistance(position) {
        const dx = this.x - position.x;
        const dy = this.y - position.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    /**
     * Gets the closest position adjacent to the given position, to this position. In the case of
     * ties, prioritizes in the following order: north, south, east, west.
     *
     * @param position  the position to get the adjacent position of that is closest to this
     *                  position
     * @return          the closest position adjacent to the given position, to this position
     */
    closestAdjacentPosition(position) {
        const northTileDistance = this.distance(new Position(position.x, position.y + 1));
        const southTileDistance = this.distance(new Position(position.x, position.y - 1));
        const eastTileDistance = this.distance(new Position(position.x + 1, position.y));
        const westTileDistance = this.distance(new Position(position.x - 1, position.y));
        const minimumDistance = Math.min(northTileDistance, southTileDistance, eastTileDistance, westTileDistance);
        if (minimumDistance === northTileDistance) {
            return new Position(position.x, position.y + 1);
        }
        if (minimumDistance === southTileDistance) {
            return new Position(position.x, position.y - 1);
        }
        if (minimumDistance === eastTileDistance) {
            return new Position(position.x + 1, position.y);
        }
        return new Position(position.x - 1, position.y);
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let position = new Position(this.x, this.y);
        position.x = this.x;
        position.y = this.y;
        return position;
    }
}
