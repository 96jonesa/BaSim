import {Position} from "./Position.js";
import {BarbarianAssault} from "./BarbarianAssault.js";
import {Character} from "./Character.js";

/**
 * Represents a Barbarian Assault player character.
 */
export abstract class Player extends Character {
    public pathQueueIndex: number = 0;
    public pathQueuePositions: Array<Position> = [];
    public shortestDistances: Array<number> = [];
    public waypoints: Array<number> = [];

    protected constructor(position: Position) {
        super(position);
    }

    /**
     * Determines the path this player must take to move to the given destination in the given
     * BarbarianAssault game, and updates this player's state to move along this path to the given
     * destination over time.
     *
     * @param barbarianAssault  the BarbarianAssault game to find the path to the given destination in
     * @param destination       the position to find the path to in the given BarbarianAssault game
     */
    public findPath(barbarianAssault: BarbarianAssault, destination: Position): void {
        for (let i: number = 0; i < barbarianAssault.map.width * barbarianAssault.map.height; i++) {
            this.shortestDistances[i] = 99999999;
            this.waypoints[i] = 0;
        }

        this.waypoints[this.position.x + this.position.y * barbarianAssault.map.width] = 99;
        this.shortestDistances[this.position.x + this.position.y * barbarianAssault.map.width] = 0;
        this.pathQueueIndex = 0;
        let pathQueueEnd: number = 0;
        this.pathQueuePositions[pathQueueEnd] = new Position(this.position.x, this.position.y);
        pathQueueEnd++;

        let currentPosition: Position;
        let foundDestination: boolean = false;

        while (this.pathQueueIndex !== pathQueueEnd) {
            currentPosition = new Position(this.pathQueuePositions[this.pathQueueIndex].x, this.pathQueuePositions[this.pathQueueIndex].y);
            this.pathQueueIndex++;

            if (currentPosition.equals(destination)) {
                foundDestination = true;
                break;
            }

            const newDistance: number = this.shortestDistances[currentPosition.x + currentPosition.y * barbarianAssault.map.width] + 1;

            let index: number = currentPosition.x - 1 + currentPosition.y * barbarianAssault.map.width;

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
            let bestDistanceStart: number = 0x7FFFFFFF;
            let bestDistanceEnd: number = 0x7FFFFFFF;
            const deviation: number = 10;

            for (let x: number = destination.x - deviation; x <= destination.x + deviation; x++) {
                for (let y: number = destination.y - deviation; y <= destination.y + deviation; y++) {
                    if (x >= 0 && y >= 0 && x < barbarianAssault.map.width && y < barbarianAssault.map.height) {
                        const distanceStart: number = this.shortestDistances[x + y * barbarianAssault.map.width];

                        if (distanceStart < 100) {
                            const distanceEnd: number = Math.max(destination.x - x) ** 2 + Math.max(destination.y - y) ** 2;

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
            const waypoint: number = this.waypoints[currentPosition.x + currentPosition.y * barbarianAssault.map.width];

            this.pathQueuePositions[this.pathQueueIndex] = new Position(currentPosition.x, currentPosition.y);
            this.pathQueueIndex++;

            if ((waypoint & 2) !== 0) {
                currentPosition.x++;
            } else if ((waypoint & 8) !== 0) {
                currentPosition.x--;
            }

            if ((waypoint & 1) !== 0) {
                currentPosition.y++;
            } else if ((waypoint & 4) !== 0) {
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
    protected move(): void {
        this.takeSteps(2);
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
    private takeSteps(steps: number): number {
        let stepsTaken: number = 0;

        while (stepsTaken < steps && this.pathQueueIndex > 0) {
            this.pathQueueIndex--;
            this.position = this.pathQueuePositions[this.pathQueueIndex];
            stepsTaken++;
        }

        return stepsTaken;
    }
}