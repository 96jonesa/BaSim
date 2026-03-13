import { Position } from "./Position.js";
import { Character } from "./Character.js";
const MOVEMENT_PRIORITY = {
    "": 0,
    "w": 1,
    "e": 2,
    "s": 3,
    "n": 4,
    "sw": 5,
    "se": 6,
    "nw": 7,
    "ne": 8,
};
/**
 * Represents a Barbarian Assault player character.
 */
export class Player extends Character {
    constructor(position) {
        super(position);
        this.pathQueueIndex = 0;
        this.pathQueuePositions = [];
        this.shortestDistances = [];
        this.waypoints = [];
        this.codeQueue = [];
        this.codeIndex = 0;
        this.arriveDelay = false;
        this.prevPosition = null;
        this.isRunning = true;
    }
    clearCodeQueue() {
        this.codeQueue = [];
        this.codeIndex = 0;
    }
    processCodeQueue(barbarianAssault) {
        if (this.codeIndex >= this.codeQueue.length) {
            return;
        }
        const action = this.codeQueue[this.codeIndex];
        if (barbarianAssault.ticks < action.waitUntil) {
            return;
        }
        const healer = this.findHealerById(barbarianAssault, action.healerId);
        if (healer === null) {
            this.codeIndex++;
            return;
        }
        if (this.pathQueueIndex === 0 || this.shouldRecalculatePath()) {
            if (this.isCardinalAdjacentTo(healer)) {
                healer.eatFood(barbarianAssault);
                this.codeIndex++;
                this.arriveDelay = true;
                this.pathQueueIndex = 0;
            }
            else {
                this.recalculateFoodPath(barbarianAssault, healer);
            }
        }
    }
    initializeFoodPath(barbarianAssault) {
        if (this.codeIndex >= this.codeQueue.length) {
            return;
        }
        const action = this.codeQueue[this.codeIndex];
        if (barbarianAssault.ticks < action.waitUntil) {
            return;
        }
        const healer = this.findHealerById(barbarianAssault, action.healerId);
        if (healer === null) {
            return;
        }
        this.recalculateFoodPath(barbarianAssault, healer);
    }
    recalculateFoodPath(barbarianAssault, healer) {
        const adjacent = this.findBestAdjacentTile(barbarianAssault, healer.position);
        this.findPath(barbarianAssault, adjacent);
    }
    shouldRecalculatePath() {
        if (this.pathQueueIndex <= 0)
            return false;
        const step1 = this.pathQueuePositions[this.pathQueueIndex - 1];
        const dx = step1.x - this.position.x;
        const dy = step1.y - this.position.y;
        let prevPos = step1;
        for (let i = this.pathQueueIndex - 2; i >= 0; i--) {
            const step = this.pathQueuePositions[i];
            if (step.x - prevPos.x !== dx || step.y - prevPos.y !== dy)
                return false;
            prevPos = step;
        }
        return true;
    }
    isCardinalAdjacentTo(healer) {
        // true_drawnTileIsAdj: player current pos adjacent to healer drawn pos
        if (healer.drawnPosition !== null && this.isCardinalAdjacentToPosition(this.position, healer.drawnPosition)) {
            return true;
        }
        // drawn_trueTileIsAdj && true_trueTileIsAdj: player prev AND current pos
        // both adjacent to healer true pos (prevents food use on the arrival tick)
        if (this.prevPosition !== null
            && this.isCardinalAdjacentToPosition(this.prevPosition, healer.position)
            && this.isCardinalAdjacentToPosition(this.position, healer.position)) {
            return true;
        }
        return false;
    }
    isCardinalAdjacentToPosition(a, b) {
        const dx = Math.abs(a.x - b.x);
        const dy = Math.abs(a.y - b.y);
        return (dx + dy) === 1;
    }
    findHealerById(barbarianAssault, healerId) {
        for (const healer of barbarianAssault.healers) {
            if (healer.id === healerId && !healer.isDying) {
                return healer;
            }
        }
        return null;
    }
    findBestAdjacentTile(barbarianAssault, target) {
        var _a;
        const candidates = [
            { tile: new Position(target.x - 1, target.y), check: (pos) => barbarianAssault.map.canMoveEast(pos) },
            { tile: new Position(target.x + 1, target.y), check: (pos) => barbarianAssault.map.canMoveWest(pos) },
            { tile: new Position(target.x, target.y - 1), check: (pos) => barbarianAssault.map.canMoveNorth(pos) },
            { tile: new Position(target.x, target.y + 1), check: (pos) => barbarianAssault.map.canMoveSouth(pos) },
        ];
        let valid = [];
        let minDistance = Infinity;
        for (const candidate of candidates) {
            if (candidate.check(candidate.tile) && barbarianAssault.map.canMoveToTile(candidate.tile)) {
                const dist = this.position.distance(candidate.tile);
                valid.push({ tile: candidate.tile, distance: dist });
                if (dist < minDistance) {
                    minDistance = dist;
                }
            }
        }
        valid = valid.filter(v => v.distance <= minDistance);
        if (valid.length === 0) {
            return this.position.clone();
        }
        if (valid.length === 1) {
            return valid[0].tile;
        }
        const savedPathQueueIndex = this.pathQueueIndex;
        const savedPathQueuePositions = this.pathQueuePositions.slice();
        const savedShortestDistances = this.shortestDistances.slice();
        const savedWaypoints = this.waypoints.slice();
        let bestTile = valid[0].tile;
        let bestWeight = Infinity;
        for (const v of valid) {
            this.findPath(barbarianAssault, v.tile);
            const idx = this.pathQueueIndex - 1;
            if (idx < 0)
                continue;
            const stepPos = this.pathQueuePositions[idx];
            let cardinalStr = "";
            if (stepPos.y < this.position.y)
                cardinalStr += "s";
            else if (stepPos.y > this.position.y)
                cardinalStr += "n";
            if (stepPos.x < this.position.x)
                cardinalStr += "w";
            else if (stepPos.x > this.position.x)
                cardinalStr += "e";
            const weight = (_a = MOVEMENT_PRIORITY[cardinalStr]) !== null && _a !== void 0 ? _a : Infinity;
            if (weight < bestWeight) {
                bestWeight = weight;
                bestTile = v.tile;
            }
        }
        this.pathQueueIndex = savedPathQueueIndex;
        this.pathQueuePositions = savedPathQueuePositions;
        this.shortestDistances = savedShortestDistances;
        this.waypoints = savedWaypoints;
        return bestTile;
    }
    /**
     * Determines the path this player must take to move to the given destination in the given
     * BarbarianAssault game, and updates this player's state to move along this path to the given
     * destination over time.
     *
     * @param barbarianAssault  the BarbarianAssault game to find the path to the given destination in
     * @param destination       the position to find the path to in the given BarbarianAssault game
     */
    findPath(barbarianAssault, destination) {
        for (let i = 0; i < barbarianAssault.map.width * barbarianAssault.map.height; i++) {
            this.shortestDistances[i] = 99999999;
            this.waypoints[i] = 0;
        }
        this.waypoints[this.position.x + this.position.y * barbarianAssault.map.width] = 99;
        this.shortestDistances[this.position.x + this.position.y * barbarianAssault.map.width] = 0;
        this.pathQueueIndex = 0;
        let pathQueueEnd = 0;
        this.pathQueuePositions[pathQueueEnd] = new Position(this.position.x, this.position.y);
        pathQueueEnd++;
        let currentPosition;
        let foundDestination = false;
        while (this.pathQueueIndex !== pathQueueEnd) {
            currentPosition = new Position(this.pathQueuePositions[this.pathQueueIndex].x, this.pathQueuePositions[this.pathQueueIndex].y);
            this.pathQueueIndex++;
            if (currentPosition.equals(destination)) {
                foundDestination = true;
                break;
            }
            const newDistance = this.shortestDistances[currentPosition.x + currentPosition.y * barbarianAssault.map.width] + 1;
            let index = currentPosition.x - 1 + currentPosition.y * barbarianAssault.map.width;
            if (currentPosition.x > 0 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136776) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x - 1, currentPosition.y);
                pathQueueEnd++;
                this.waypoints[index] = 2;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x + 1 + currentPosition.y * barbarianAssault.map.width;
            if (currentPosition.x < barbarianAssault.map.width - 1 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136896) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x + 1, currentPosition.y);
                pathQueueEnd++;
                this.waypoints[index] = 8;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x + (currentPosition.y - 1) * barbarianAssault.map.width;
            if (currentPosition.y > 0 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136770) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x, currentPosition.y - 1);
                pathQueueEnd++;
                this.waypoints[index] = 1;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x + (currentPosition.y + 1) * barbarianAssault.map.width;
            if (currentPosition.y < barbarianAssault.map.height - 1 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136800) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x, currentPosition.y + 1);
                pathQueueEnd++;
                this.waypoints[index] = 4;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x - 1 + (currentPosition.y - 1) * barbarianAssault.map.width;
            if (currentPosition.x > 0 && currentPosition.y > 0 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136782) === 0
                && (barbarianAssault.map.map[currentPosition.x - 1 + currentPosition.y * barbarianAssault.map.width] & 19136776) === 0
                && (barbarianAssault.map.map[currentPosition.x + (currentPosition.y - 1) * barbarianAssault.map.width] & 19136770) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x - 1, currentPosition.y - 1);
                pathQueueEnd++;
                this.waypoints[index] = 3;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x + 1 + (currentPosition.y - 1) * barbarianAssault.map.width;
            if (currentPosition.x < barbarianAssault.map.width - 1 && currentPosition.y > 0 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136899) === 0
                && (barbarianAssault.map.map[currentPosition.x + 1 + currentPosition.y * barbarianAssault.map.width] & 19136896) === 0
                && (barbarianAssault.map.map[currentPosition.x + (currentPosition.y - 1) * barbarianAssault.map.width] & 19136770) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x + 1, currentPosition.y - 1);
                pathQueueEnd++;
                this.waypoints[index] = 9;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x - 1 + (currentPosition.y + 1) * barbarianAssault.map.width;
            if (currentPosition.x > 0 && currentPosition.y < barbarianAssault.map.height - 1 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136824) === 0
                && (barbarianAssault.map.map[currentPosition.x - 1 + currentPosition.y * barbarianAssault.map.width] & 19136776) === 0
                && (barbarianAssault.map.map[currentPosition.x + (currentPosition.y + 1) * barbarianAssault.map.width] & 19136800) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x - 1, currentPosition.y + 1);
                pathQueueEnd++;
                this.waypoints[index] = 6;
                this.shortestDistances[index] = newDistance;
            }
            index = currentPosition.x + 1 + (currentPosition.y + 1) * barbarianAssault.map.width;
            if (currentPosition.x < barbarianAssault.map.width - 1 && currentPosition.y < barbarianAssault.map.height - 1 && this.waypoints[index] === 0 && (barbarianAssault.map.map[index] & 19136992) === 0
                && (barbarianAssault.map.map[currentPosition.x + 1 + currentPosition.y * barbarianAssault.map.width] & 19136896) === 0
                && (barbarianAssault.map.map[currentPosition.x + (currentPosition.y + 1) * barbarianAssault.map.width] & 19136800) === 0) {
                this.pathQueuePositions[pathQueueEnd] = new Position(currentPosition.x + 1, currentPosition.y + 1);
                pathQueueEnd++;
                this.waypoints[index] = 12;
                this.shortestDistances[index] = newDistance;
            }
        }
        if (!foundDestination) {
            let bestDistanceStart = 0x7FFFFFFF;
            let bestDistanceEnd = 0x7FFFFFFF;
            const deviation = 10;
            for (let x = destination.x - deviation; x <= destination.x + deviation; x++) {
                for (let y = destination.y - deviation; y <= destination.y + deviation; y++) {
                    if (x >= 0 && y >= 0 && x < barbarianAssault.map.width && y < barbarianAssault.map.height) {
                        const distanceStart = this.shortestDistances[x + y * barbarianAssault.map.width];
                        if (distanceStart < 100) {
                            const distanceEnd = Math.max(destination.x - x) ** 2 + Math.max(destination.y - y) ** 2;
                            if (distanceEnd < bestDistanceEnd || (distanceEnd === bestDistanceEnd && distanceStart < bestDistanceStart)) {
                                bestDistanceStart = distanceStart;
                                bestDistanceEnd = distanceEnd;
                                currentPosition = new Position(x, y);
                                foundDestination = true;
                            }
                        }
                    }
                }
            }
            if (!foundDestination) {
                this.pathQueueIndex = 0;
                return;
            }
        }
        this.pathQueueIndex = 0;
        while (!currentPosition.equals(this.position)) {
            const waypoint = this.waypoints[currentPosition.x + currentPosition.y * barbarianAssault.map.width];
            this.pathQueuePositions[this.pathQueueIndex] = new Position(currentPosition.x, currentPosition.y);
            this.pathQueueIndex++;
            if ((waypoint & 2) !== 0) {
                currentPosition.x++;
            }
            else if ((waypoint & 8) !== 0) {
                currentPosition.x--;
            }
            if ((waypoint & 1) !== 0) {
                currentPosition.y++;
            }
            else if ((waypoint & 4) !== 0) {
                currentPosition.y--;
            }
        }
    }
    /**
     * This player takes up to two steps (as many as possible) in its path to
     * its destination.
     *
     * @private
     */
    move() {
        this.takeSteps(this.isRunning ? 2 : 1);
    }
    /**
     * This player takes up to the given number of steps (as many as possible) in
     * its path to its destination.
     *
     * @param steps the maximum number of steps for this player to take in
     *              its path to its destination
     * @return      the number of steps taken
     * @private
     */
    takeSteps(steps) {
        let stepsTaken = 0;
        while (stepsTaken < steps && this.pathQueueIndex > 0) {
            this.pathQueueIndex--;
            this.position = this.pathQueuePositions[this.pathQueueIndex];
            stepsTaken++;
        }
        return stepsTaken;
    }
}
