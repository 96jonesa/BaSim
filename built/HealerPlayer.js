import { Player } from "./Player.js";
/**
 * Represents a Barbarian Assault healer player.
 */
export class HealerPlayer extends Player {
    constructor(position) {
        super(position);
        this.codeQueue = [];
        this.codeIndex = 0;
    }
    /**
     * @inheritDoc
     */
    tick(barbarianAssault) {
        if (this.codeQueue.length > 0) {
            this.processCodeQueue(barbarianAssault);
        }
        this.move();
    }
    processCodeQueue(barbarianAssault) {
        if (this.codeIndex >= this.codeQueue.length) {
            return;
        }
        const action = this.codeQueue[this.codeIndex];
        // Wait until the specified tick
        if (barbarianAssault.ticks < action.waitUntil) {
            return;
        }
        // Skip dead healers
        const healer = this.findHealer(barbarianAssault, action.healerId);
        if (healer === null) {
            this.codeIndex++;
            return;
        }
        // Check if adjacent to healer
        const adjacent = this.position.closestAdjacentPosition(healer.position);
        if (this.position.equals(adjacent)) {
            healer.eatFood(barbarianAssault);
            this.codeIndex++;
            // Immediately target next action if there is one
            if (this.codeIndex < this.codeQueue.length) {
                const nextAction = this.codeQueue[this.codeIndex];
                if (barbarianAssault.ticks >= nextAction.waitUntil) {
                    const nextHealer = this.findHealer(barbarianAssault, nextAction.healerId);
                    if (nextHealer !== null) {
                        const nextAdj = this.position.closestAdjacentPosition(nextHealer.position);
                        this.findPath(barbarianAssault, nextAdj);
                    }
                }
            }
        }
        else {
            // Pathfind to adjacent tile of target healer
            this.findPath(barbarianAssault, adjacent);
        }
    }
    findHealer(barbarianAssault, healerId) {
        for (const healer of barbarianAssault.healers) {
            if (healer.id === healerId && !healer.isDying) {
                return healer;
            }
        }
        return null;
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let healerPlayer = new HealerPlayer(this.position);
        healerPlayer.position = this.position === null ? null : this.position.clone();
        healerPlayer.codeQueue = this.codeQueue.map(a => a.clone());
        healerPlayer.codeIndex = this.codeIndex;
        return healerPlayer;
    }
}
