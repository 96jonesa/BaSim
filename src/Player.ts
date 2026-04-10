import {Position} from "./Position.js";
import {BarbarianAssault} from "./BarbarianAssault.js";
import {Character} from "./Character.js";
import {HealerCodeAction} from "./HealerCodeAction.js";
import {HealerPenance} from "./HealerPenance.js";
import {SeedType} from "./SeedType.js";

const MOVEMENT_PRIORITY: Record<string, number> = {
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
export abstract class Player extends Character {
    public checkpoints: Array<Position> = [];
    public checkpointIndex: number = 0;
    public pathDestination: Position = null;
    public codeQueue: Array<HealerCodeAction> = [];
    public codeIndex: number = 0;
    public arriveDelay: boolean = false;
    public prevPosition: Position = null;
    public isRunning: boolean = true;
    public pendingSeed: SeedType = null;
    public seedMovedThisTick: boolean = false;
    public preSeedPosition: Position = null;
    public seedMovedToPosition: Position = null;
    public repeatSeedType: SeedType = null;
    public redXHealerId: number = null;
    public isRedXPath: boolean = false;
    public phased: boolean = false;
    public pathStepPositions: Array<Position> = [];

    // Working arrays for findPath BFS — not meaningful outside findPath
    private pathQueuePositions: Array<Position> = [];
    private pathQueueIndex: number = 0;
    private shortestDistances: Array<number> = [];
    private waypoints: Array<number> = [];

    protected constructor(position: Position) {
        super(position);
    }

    public clearCodeQueue(): void {
        this.codeQueue = [];
        this.codeIndex = 0;
    }

    public clearPath(): void {
        this.pathDestination = null;
        this.isRedXPath = false;
        this.checkpoints = [];
        this.checkpointIndex = 0;
    }

    public processCodeQueue(barbarianAssault: BarbarianAssault): void {
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

        if (this.isCardinalAdjacentTo(healer)) {
            healer.eatFood(barbarianAssault);
            this.codeIndex++;
            this.arriveDelay = true;
            this.clearPath();
            this.initializeFoodPath(barbarianAssault);
        } else if (this.pathDestination === null || this.shouldRecalculatePath()) {
            this.recalculateFoodPath(barbarianAssault, healer);
        }
    }

    public initializeFoodPath(barbarianAssault: BarbarianAssault): void {
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

    private recalculateFoodPath(barbarianAssault: BarbarianAssault, healer: HealerPenance): void {
        const adjacent = this.findBestAdjacentTile(barbarianAssault, healer.position);
        this.findPath(barbarianAssault, adjacent);
    }

    private shouldRecalculatePath(): boolean {
        return this.checkpointIndex >= this.checkpoints.length;
    }

    private isCardinalAdjacentTo(healer: HealerPenance): boolean {
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

    private isCardinalAdjacentToPosition(a: Position, b: Position): boolean {
        const dx = Math.abs(a.x - b.x);
        const dy = Math.abs(a.y - b.y);
        return (dx + dy) === 1;
    }

    private findHealerById(barbarianAssault: BarbarianAssault, healerId: number): HealerPenance | null {
        for (const healer of barbarianAssault.healers) {
            if (healer.id === healerId && !healer.isDying) {
                return healer;
            }
        }
        return null;
    }

    public findBestAdjacentTile(barbarianAssault: BarbarianAssault, target: Position): Position {
        const candidates: Array<{ tile: Position; check: (pos: Position) => boolean }> = [
            { tile: new Position(target.x - 1, target.y), check: (pos) => barbarianAssault.map.canMoveEast(pos) },
            { tile: new Position(target.x + 1, target.y), check: (pos) => barbarianAssault.map.canMoveWest(pos) },
            { tile: new Position(target.x, target.y - 1), check: (pos) => barbarianAssault.map.canMoveNorth(pos) },
            { tile: new Position(target.x, target.y + 1), check: (pos) => barbarianAssault.map.canMoveSouth(pos) },
        ];

        const valid: Array<Position> = [];

        for (const candidate of candidates) {
            if (candidate.check(candidate.tile) && barbarianAssault.map.canMoveToTile(candidate.tile)) {
                valid.push(candidate.tile);
            }
        }

        if (valid.length === 0) {
            return this.position.clone();
        }

        if (valid.length === 1) {
            return valid[0];
        }

        const savedCheckpoints = this.checkpoints.map(p => p.clone());
        const savedCheckpointIndex = this.checkpointIndex;
        const savedPathDestination = this.pathDestination === null ? null : this.pathDestination.clone();
        const savedIsRedXPath = this.isRedXPath;

        let bestTile: Position = valid[0];
        let bestSteps: number = Infinity;
        let bestWeight: number = Infinity;

        for (const tile of valid) {
            this.findPath(barbarianAssault, tile);
            const steps = this.shortestDistances[tile.x + tile.y * barbarianAssault.map.width];

            if (steps > bestSteps) continue;

            const firstTarget = this.checkpoints.length > 0 ? this.checkpoints[0] : this.pathDestination;
            let weight = Infinity;
            if (firstTarget !== null) {
                let cardinalStr = "";
                if (firstTarget.y < this.position.y) cardinalStr += "s";
                else if (firstTarget.y > this.position.y) cardinalStr += "n";
                if (firstTarget.x < this.position.x) cardinalStr += "w";
                else if (firstTarget.x > this.position.x) cardinalStr += "e";
                weight = MOVEMENT_PRIORITY[cardinalStr] ?? Infinity;
            }

            if (steps < bestSteps || weight < bestWeight) {
                bestSteps = steps;
                bestWeight = weight;
                bestTile = tile;
            }
        }

        this.checkpoints = savedCheckpoints;
        this.checkpointIndex = savedCheckpointIndex;
        this.pathDestination = savedPathDestination;
        this.isRedXPath = savedIsRedXPath;

        return bestTile;
    }

    /**
     * Determines the path this player must take to move to the given destination in the given
     * BarbarianAssault game, and updates this player's checkpoints and destination accordingly.
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
                this.clearPath();
                return;
            }
        }

        // Traceback: build path from destination back to player position
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

        // Extract checkpoints from path
        // Path is in pathQueuePositions[0..pathQueueIndex-1] in reverse order:
        //   [pathQueueIndex-1] = first step, [0] = destination
        if (this.pathQueueIndex === 0) {
            // Already at destination
            this.pathDestination = destination.clone();
            this.checkpoints = [];
            this.checkpointIndex = 0;
            return;
        }

        this.pathDestination = new Position(this.pathQueuePositions[0].x, this.pathQueuePositions[0].y);
        this.checkpoints = [];
        this.checkpointIndex = 0;

        if (this.pathQueueIndex >= 2) {
            // Walk path forward: from first step (pathQueueIndex-1) to destination (0)
            let prevPos = this.position;
            let prevDx = this.pathQueuePositions[this.pathQueueIndex - 1].x - prevPos.x;
            let prevDy = this.pathQueuePositions[this.pathQueueIndex - 1].y - prevPos.y;
            prevPos = this.pathQueuePositions[this.pathQueueIndex - 1];

            for (let i = this.pathQueueIndex - 2; i >= 0; i--) {
                const cur = this.pathQueuePositions[i];
                const dx = cur.x - prevPos.x;
                const dy = cur.y - prevPos.y;

                if (dx !== prevDx || dy !== prevDy) {
                    // Direction changed — prevPos is a checkpoint
                    this.checkpoints.push(new Position(prevPos.x, prevPos.y));
                }

                prevDx = dx;
                prevDy = dy;
                prevPos = cur;
            }
        }
    }

    /**
     * This player takes up to two steps (running) or one step (walking)
     * toward its next checkpoint or destination using healer-penance-style movement.
     */
    protected move(barbarianAssault: BarbarianAssault): void {
        this.pathStepPositions = [];
        const steps = (this.isRunning && !this.seedMovedThisTick) ? 2 : 1;
        for (let s = 0; s < steps; s++) {
            if (this.pathDestination === null) break;

            const target = this.checkpointIndex < this.checkpoints.length
                ? this.checkpoints[this.checkpointIndex]
                : this.pathDestination;

            if (this.position.equals(target) && target === this.pathDestination) {
                this.clearPath();
                break;
            }

            this.stepToward(barbarianAssault, target);
            this.pathStepPositions.push(this.position.clone());

            if (this.checkpointIndex < this.checkpoints.length && this.position.equals(this.checkpoints[this.checkpointIndex])) {
                this.checkpointIndex++;
            }

            if (this.checkpointIndex >= this.checkpoints.length && this.position.equals(this.pathDestination)) {
                this.clearPath();
                break;
            }
        }
    }

    /**
     * Takes a single step toward the target using healer-penance-style movement
     * (move X first, then Y), without penance blocking checks.
     */
    private stepToward(barbarianAssault: BarbarianAssault, target: Position): void {
        const startX = this.position.x;

        if (target.x > startX) {
            if (barbarianAssault.map.canMoveEast(new Position(startX, this.position.y))) {
                this.position.x++;
            }
        } else if (target.x < startX) {
            if (barbarianAssault.map.canMoveWest(new Position(startX, this.position.y))) {
                this.position.x--;
            }
        }

        if (target.y > this.position.y) {
            if (barbarianAssault.map.canMoveNorth(new Position(startX, this.position.y)) && barbarianAssault.map.canMoveNorth(new Position(this.position.x, this.position.y))) {
                this.position.y++;
            }
        } else if (target.y < this.position.y) {
            if (barbarianAssault.map.canMoveSouth(new Position(startX, this.position.y)) && barbarianAssault.map.canMoveSouth(new Position(this.position.x, this.position.y))) {
                this.position.y--;
            }
        }
    }
}
