import { Penance } from "./Penance.js";
import { Position } from "./Position.js";
import { Player } from "./Player.js";
import { HealerTargetType } from "./HealerTargetType.js";
import { RunnerPenance } from "./RunnerPenance.js";
/**
 * Represents a Barbarian Assault healer penance.
 */
export class HealerPenance extends Penance {
    constructor(position, maxHealth, spawnTick, id) {
        super(position);
        this.target = null;
        this.previousTargetType = null;
        this.sprayTimer = 0;
        this.isDying = false;
        this.despawnCountdown = null;
        this.isPoisoned = false;
        this.poisonDamage = 4;
        this.poisonTickCount = 0;
        this.poisonHitsplat = false;
        this.regenTimer = -1;
        this.spawnPosition = position.clone();
        this.maxHealth = maxHealth;
        this.health = maxHealth;
        this.spawnTick = spawnTick;
        this.id = id;
    }
    /**
     * @inheritDoc
     */
    tick(barbarianAssault) {
        this.poisonHitsplat = false;
        this.regenTimer++;
        if (this.regenTimer % 100 === 0) {
            if (this.health < this.maxHealth) {
                this.health++;
            }
        }
        this.applyPoisonDamage(barbarianAssault);
        this.processDeath(barbarianAssault);
        if (this.isDying) {
            return;
        }
        this.updateTargetAndPursueOrMove(barbarianAssault);
    }
    /**
     * Updates this healer penance's target then pursues its target if it has one, or attempts to
     * move randomly if it does not have one.
     *
     * @param barbarianAssault  the BarbarianAssault game in which this healer penance moves and
     *                          interacts with targets
     * @private
     */
    updateTargetAndPursueOrMove(barbarianAssault) {
        if (this.target === null && this.previousTargetType === null) {
            this.tryToTargetPlayer(barbarianAssault);
        }
        if (this.target === null && this.previousTargetType !== null) {
            this.sprayTimer++;
            if (this.previousTargetType === HealerTargetType.RUNNER && this.sprayTimer > 2) {
                this.tryToTargetPlayer(barbarianAssault);
            }
            else if (this.previousTargetType === HealerTargetType.PLAYER && (this.sprayTimer + 1) % 3 === 0 && this.sprayTimer > 4) {
                this.tryToTargetRunner(barbarianAssault);
            }
            if (this.target === null) {
                this.tryToSetRandomDestination();
                this.move(barbarianAssault);
            }
            else {
                this.pursueTarget(barbarianAssault);
            }
        }
        else if (this.target instanceof Player) {
            this.pursueTarget(barbarianAssault);
        }
        else if (this.target instanceof RunnerPenance) {
            if (this.target.despawnCountdown === 0) {
                console.log(barbarianAssault.ticks + ": retargeting");
                this.previousTargetType = HealerTargetType.RUNNER;
                this.sprayTimer = 0;
                this.tryToTargetPlayer(barbarianAssault);
                if (this.target === null) {
                    this.tryToSetRandomDestination();
                    this.move(barbarianAssault);
                }
                else {
                    this.pursueTarget(barbarianAssault);
                }
            }
            else {
                this.pursueTarget(barbarianAssault);
            }
        }
    }
    /**
     * Applies any applicable damage if this healer penance is currently poisoned.
     *
     * @param barbarianAssault  the BarbarianAssault game according to which the applicability
     *                          of poison damage for this healer penance is determined
     * @private
     */
    applyPoisonDamage(barbarianAssault) {
        if (this.isDying) {
            return;
        }
        if (this.isPoisoned && barbarianAssault.ticks - this.spawnTick >= 5) {
            if (barbarianAssault.ticks - this.lastPoisonTick >= 5) {
                this.health = Math.max(0, this.health - this.poisonDamage);
                this.poisonHitsplat = true;
                this.lastPoisonTick = barbarianAssault.ticks;
                this.poisonTickCount++;
            }
            if (this.poisonTickCount === 5) {
                this.poisonDamage--;
                this.poisonTickCount = 0;
            }
            if (this.poisonDamage <= 0) {
                this.isPoisoned = false;
            }
        }
    }
    /**
     * This healer penance eats a poison food, taking damage and becoming fully poisoned.
     *
     * @param barbarianAssault  the BarbarianAssault game according to which the timer for
     *                          the poison induced by the eaten food is determined
     */
    eatFood(barbarianAssault) {
        if (this.isDying) {
            return;
        }
        this.health = Math.max(0, this.health - 4);
        this.poisonDamage = 4;
        this.poisonTickCount = 0;
        if (!this.isPoisoned) {
            this.isPoisoned = true;
            if (barbarianAssault.ticks - this.spawnTick >= 5) {
                this.spawnTick = barbarianAssault.ticks;
            }
        }
        this.poisonHitsplat = true;
    }
    /**
     * Handles the death of this healer penance if its health is non-positive: This healer penance
     * is marked as dying; if it is the first tick its health has been non-positive, then the
     * given BarbarianAssault game is updated to consider it both killed and not alive; and if it
     * is the third tick its health has been non-positive, then the given BarbariansAssault game
     * is updated to remove it.
     *
     * @param barbarianAssault  the BarbarianAssault game to update as this healer penance dies
     * @private
     */
    processDeath(barbarianAssault) {
        if (this.health > 0) {
            return;
        }
        this.isDying = true;
        if (this.despawnCountdown === null) {
            this.despawnCountdown = 2;
            barbarianAssault.healersAlive--;
            barbarianAssault.healersKilled++;
        }
        else {
            this.despawnCountdown--;
        }
        if (this.despawnCountdown == 0) {
            barbarianAssault.healersToRemove.push(this);
        }
    }
    /**
     * This healer penance targets a random in-range Player.
     *
     * @param barbarianAssault  the BarbarianAssault game for this healer penance to attempt to
     *                          target a Player in
     * @private
     */
    tryToTargetPlayer(barbarianAssault) {
        const players = [
            barbarianAssault.collectorPlayer,
            barbarianAssault.defenderPlayer,
            barbarianAssault.mainAttackerPlayer,
            barbarianAssault.secondAttackerPlayer,
            barbarianAssault.healerPlayer
        ];
        const candidates = players.filter((player) => {
            return barbarianAssault.map.hasLineOfSight(this.position, player.position, 15);
        });
        if (candidates.length > 0) {
            this.target = candidates[Math.floor(Math.random() * candidates.length)];
            return;
        }
        this.target = null;
    }
    /**
     * This healer penance targets a random in-range RunnerPenance.
     *
     * @param barbarianAssault  the BarbarianAssault game for this healer penance to attempt to
     *                          target a RunnerPenance in
     * @private
     */
    tryToTargetRunner(barbarianAssault) {
        const candidates = barbarianAssault.runners.filter((runner) => {
            return barbarianAssault.map.hasLineOfSight(this.position, runner.position, 5) && !runner.isDying;
        });
        if (candidates.length > 0) {
            this.target = candidates[Math.floor(Math.random() * candidates.length)];
            return;
        }
        this.target = null;
    }
    /**
     * If this healer penance did not spray a Player on the previous tick, then sets its
     * destination to a random tile in a square of side length 121 tiles centered at the
     * tile it spawned at.
     *
     * @private
     */
    tryToSetRandomDestination() {
        if (this.sprayTimer === 1 && this.previousTargetType === HealerTargetType.PLAYER) {
            return;
        }
        if (Math.floor(Math.random() * 8) === 0) {
            const newDestinationX = this.spawnPosition.x - 60 + Math.floor(Math.random() * 121);
            const newDestinationY = this.spawnPosition.y - 60 + Math.floor(Math.random() * 121);
            this.destination = new Position(newDestinationX, newDestinationY);
        }
    }
    /**
     * If this healer penance has a target, then it attempts to spray its target. If it fails to
     * spray its target, then it moves toward its target and attempts to spray its target again.
     *
     * @param barbarianAssault  the BarbarianAssault game in which this healer penance moves
     *                          and attempts to spray its target in
     * @private
     */
    pursueTarget(barbarianAssault) {
        if (this.target === null) {
            return;
        }
        if (!this.tryToSpray(barbarianAssault)) {
            this.destination = this.position.closestAdjacentPosition(this.target.position);
            this.move(barbarianAssault);
            this.tryToSpray(barbarianAssault);
        }
    }
    /**
     * This healer penance attempts to spray its target.
     *
     * @param barbarianAssault  the Barbarian Assault game in which this healer penance attempts
     *                          to spray its target
     * @private
     */
    tryToSpray(barbarianAssault) {
        if (this.target instanceof RunnerPenance) {
            return this.tryToSprayRunner(barbarianAssault);
        }
        else if (this.target instanceof Player) {
            return this.tryToSprayPlayer(barbarianAssault);
        }
        return false;
    }
    /**
     * This healer penance attempts to spray its (RunnerPenance) target.
     *
     * @param barbarianAssault  the Barbarian Assault game in which this healer penance attempts
     *                          to spray its (RunnerPenance) target
     * @private
     */
    tryToSprayRunner(barbarianAssault) {
        if (this.position.equals(this.position.closestAdjacentPosition(this.target.position))) {
            this.previousTargetType = HealerTargetType.RUNNER;
            this.target = null;
            this.sprayTimer = 0;
            return true;
        }
        return false;
    }
    /**
     * This healer penance attempts to spray its (Player) target.
     *
     * @param barbarianAssault  the Barbarian Assault game in which this healer penance attempts
     *                          to spray its (Player) target
     * @private
     */
    tryToSprayPlayer(barbarianAssault) {
        if (this.position.equals(this.position.closestAdjacentPosition(this.target.position)) && barbarianAssault.map.hasLineOfSight(this.position, this.target.position, 15)) {
            this.previousTargetType = HealerTargetType.PLAYER;
            this.target = null;
            this.sprayTimer = 0;
            return true;
        }
        return false;
    }
    /**
     * This healer penance takes up to one step (if possible) in its path to its destination;
     *
     * @param barbarianAssault  the BarbarianAssault game in which this healer penance moves
     * @private
     */
    move(barbarianAssault) {
        const startX = this.position.x;
        if (this.destination.x > startX) {
            if (!barbarianAssault.tileBlocksPenance(new Position(startX + 1, this.position.y)) && barbarianAssault.map.canMoveEast(new Position(startX, this.position.y))) {
                this.position.x++;
            }
        }
        else if (this.destination.x < startX) {
            if (!barbarianAssault.tileBlocksPenance(new Position(startX - 1, this.position.y)) && barbarianAssault.map.canMoveWest(new Position(startX, this.position.y))) {
                this.position.x--;
            }
        }
        if (this.destination.y > this.position.y) {
            if (!barbarianAssault.tileBlocksPenance(new Position(startX, this.position.y + 1)) && !barbarianAssault.tileBlocksPenance(new Position(this.position.x, this.position.y + 1)) && barbarianAssault.map.canMoveNorth(new Position(startX, this.position.y)) && barbarianAssault.map.canMoveNorth(new Position(this.position.x, this.position.y))) {
                this.position.y++;
            }
        }
        else if (this.destination.y < this.position.y) {
            if (!barbarianAssault.tileBlocksPenance(new Position(startX, this.position.y - 1)) && !barbarianAssault.tileBlocksPenance(new Position(this.position.x, this.position.y - 1)) && barbarianAssault.map.canMoveSouth(new Position(startX, this.position.y)) && barbarianAssault.map.canMoveSouth(new Position(this.position.x, this.position.y))) {
                this.position.y--;
            }
        }
    }
    /**
     * @inheritDoc
     */
    clone() {
        let healerPenance = new HealerPenance(this.position, this.maxHealth, this.spawnTick, this.id);
        healerPenance.spawnPosition = this.spawnPosition === null ? null : this.spawnPosition.clone();
        healerPenance.target = this.target === null ? null : this.target.clone();
        healerPenance.previousTargetType = this.previousTargetType;
        healerPenance.sprayTimer = this.sprayTimer;
        healerPenance.isDying = this.isDying;
        healerPenance.despawnCountdown = this.despawnCountdown;
        healerPenance.id = this.id;
        healerPenance.health = this.health;
        healerPenance.maxHealth = this.maxHealth;
        healerPenance.spawnTick = this.spawnTick;
        healerPenance.isPoisoned = this.isPoisoned;
        healerPenance.poisonDamage = this.poisonDamage;
        healerPenance.poisonTickCount = this.poisonTickCount;
        healerPenance.poisonHitsplat = this.poisonHitsplat;
        healerPenance.regenTimer = this.regenTimer;
        return healerPenance;
    }
}
