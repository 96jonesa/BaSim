import { FoodType } from "./FoodType.js";
import { Position } from "./Position.js";
import { BarbarianAssaultMap } from "./BarbarianAssaultMap.js";
import { RunnerPenance } from "./RunnerPenance.js";
import { DefenderPlayer } from "./DefenderPlayer.js";
import { RunnerPenanceRng } from "./RunnerPenanceRng.js";
import { CollectorPlayer } from "./CollectorPlayer.js";
import { AttackerPlayer } from "./AttackerPlayer.js";
import { HealerPlayer } from "./HealerPlayer.js";
import { HealerPenance } from "./HealerPenance.js";
/**
 * Represents a game of Barbarian Assault: holds state information and exposes functions for
 * progressing the game state.
 */
export class BarbarianAssault {
    constructor(wave, requireRepairs, requireLogs, infiniteFood, runnerMovements, defenderLevel, mainAttackerCommands, secondAttackerCommands, healerCommands, collectorCommands, defenderCommands, foodCalls) {
        this.ticks = 0;
        this.eastTrapCharges = 2;
        this.westTrapCharges = 2;
        this.northwestLogsArePresent = true;
        this.southeastLogsArePresent = true;
        this.eastTrapPosition = new Position(45, 26);
        this.westTrapPosition = new Position(15, 25);
        this.runnersToRemove = [];
        this.runnersAlive = 0;
        this.runnersKilled = 0;
        this.healersToRemove = [];
        this.healers = [];
        this.healersAlive = 0;
        this.healersKilled = 0;
        this.runners = [];
        this.runnerMovementsIndex = 0;
        this.currentRunnerId = 1;
        this.currentHealerId = 1;
        this.foodCallsIndex = 0;
        this.wave = wave;
        this.requireRepairs = requireRepairs;
        this.requireLogs = requireLogs;
        this.infiniteFood = infiniteFood;
        this.runnerMovements = runnerMovements;
        this.defenderLevel = defenderLevel;
        this.mainAttackerCommands = mainAttackerCommands;
        this.secondAttackerCommands = secondAttackerCommands;
        this.healerCommands = healerCommands;
        this.collectorCommands = collectorCommands;
        this.defenderCommands = defenderCommands;
        this.foodCalls = foodCalls;
        switch (wave) {
            case 1:
                this.maxRunnersAlive = 2;
                this.totalRunners = 2;
                this.maxHealersAlive = 2;
                this.totalHealers = 2;
                this.maxHealerHealth = 27;
                break;
            case 2:
                this.maxRunnersAlive = 2;
                this.totalRunners = 3;
                this.maxHealersAlive = 3;
                this.totalHealers = 3;
                this.maxHealerHealth = 32;
                break;
            case 3:
                this.maxRunnersAlive = 2;
                this.totalRunners = 4;
                this.maxHealersAlive = 2;
                this.totalHealers = 3;
                this.maxHealerHealth = 37;
                break;
            case 4:
                this.maxRunnersAlive = 3;
                this.totalRunners = 4;
                this.maxHealersAlive = 3;
                this.totalHealers = 4;
                this.maxHealerHealth = 43;
                break;
            case 5:
                this.maxRunnersAlive = 4;
                this.totalRunners = 5;
                this.maxHealersAlive = 4;
                this.totalHealers = 5;
                this.maxHealerHealth = 49;
                break;
            case 6:
                this.maxRunnersAlive = 4;
                this.totalRunners = 6;
                this.maxHealersAlive = 4;
                this.totalHealers = 6;
                this.maxHealerHealth = 55;
                break;
            case 7:
            case 10:
                this.maxRunnersAlive = 5;
                this.totalRunners = 6;
                this.maxHealersAlive = 4;
                this.totalHealers = 7;
                this.maxHealerHealth = 60;
                break;
            case 8:
                this.maxRunnersAlive = 5;
                this.totalRunners = 7;
                this.maxHealersAlive = 5;
                this.totalHealers = 7;
                this.maxHealerHealth = 67;
                break;
            case 9:
                this.maxRunnersAlive = 5;
                this.totalRunners = 9;
                this.maxHealersAlive = 6;
                this.totalHealers = 8;
                this.maxHealerHealth = 76;
                break;
        }
        if (wave === 10) {
            this.northwestLogsPosition = new Position(29, 39);
            this.southeastLogsPosition = new Position(30, 38);
            this.collectorPlayer = new CollectorPlayer(new Position(24, 8));
            this.defenderPlayer = new DefenderPlayer(new Position(28, 8));
            this.mainAttackerPlayer = new AttackerPlayer(new Position(26, 10));
            this.secondAttackerPlayer = new AttackerPlayer(new Position(25, 9));
            this.healerPlayer = new HealerPlayer(new Position(27, 9));
        }
        else {
            this.northwestLogsPosition = new Position(28, 39);
            this.southeastLogsPosition = new Position(29, 38);
            this.collectorPlayer = new CollectorPlayer(new Position(29, 8));
            this.defenderPlayer = new DefenderPlayer(new Position(33, 8));
            this.mainAttackerPlayer = new AttackerPlayer(new Position(31, 10));
            this.secondAttackerPlayer = new AttackerPlayer(new Position(30, 9));
            this.healerPlayer = new HealerPlayer(new Position(32, 9));
        }
        this.map = new BarbarianAssaultMap(wave);
        this.changeDefenderFoodCall();
    }
    /**
     * Progresses the game state by a single tick.
     */
    tick() {
        this.ticks++;
        console.log(this.ticks);
        this.runnersToRemove.length = 0;
        this.tickPenance();
        this.removePenance();
        if (this.ticks > 1 && this.ticks % 10 === 1) {
            this.northwestLogsArePresent = true;
            this.southeastLogsArePresent = true;
        }
        if (this.ticks > 2 && this.ticks % 50 === 2) {
            this.changeDefenderFoodCall();
        }
        if (this.ticks > 1 && this.ticks % 10 === 1 && this.runnersAlive < this.maxRunnersAlive && this.runnersKilled + this.runnersAlive < this.totalRunners) {
            this.spawnRunner();
        }
        if (this.ticks > 1 && this.ticks % 10 === 1 && this.healersAlive < this.maxHealersAlive && this.healersKilled + this.healersAlive < this.totalHealers) {
            this.spawnHealer();
        }
        this.tickPlayers();
        this.findPlayerPaths();
    }
    /**
     * Progresses each player's state by one tick.
     *
     * @private
     */
    tickPlayers() {
        this.defenderPlayer.tick(this);
        this.collectorPlayer.tick(this);
        this.mainAttackerPlayer.tick(this);
        this.secondAttackerPlayer.tick(this);
        this.healerPlayer.tick(this);
    }
    /**
     * Finds and updates the paths of each non-defender player according to its pre-specified
     * commands.
     *
     * @private
     */
    findPlayerPaths() {
        if (this.mainAttackerCommands.has(this.ticks)) {
            this.mainAttackerPlayer.findPath(this, this.mainAttackerCommands.get(this.ticks).clone());
        }
        if (this.secondAttackerCommands.has(this.ticks)) {
            this.secondAttackerPlayer.findPath(this, this.secondAttackerCommands.get(this.ticks).clone());
        }
        if (this.healerCommands.has(this.ticks)) {
            this.healerPlayer.findPath(this, this.healerCommands.get(this.ticks).clone());
        }
        if (this.collectorCommands.has(this.ticks)) {
            this.collectorPlayer.findPath(this, this.collectorCommands.get(this.ticks).clone());
        }
        if (this.defenderCommands.has(this.ticks)) {
            this.defenderPlayer.findPath(this, this.defenderCommands.get(this.ticks).clone());
        }
    }
    /**
     * Spawns a RunnerPenance.
     *
     * @private
     */
    spawnRunner() {
        let movements;
        if (this.runnerMovements.length > this.runnerMovementsIndex) {
            movements = this.runnerMovements[this.runnerMovementsIndex];
            this.runnerMovementsIndex++;
        }
        else {
            movements = "";
        }
        if (this.wave === 10) {
            this.runners.push(new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(movements), this.currentRunnerId, this.defenderLevel < 2 ? 4 : 5));
        }
        else {
            this.runners.push(new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(movements), this.currentRunnerId, this.defenderLevel < 2 ? 4 : 5));
        }
        this.currentRunnerId++;
        this.runnersAlive++;
    }
    /**
     * Spawns a HealerPenance.
     *
     * @private
     */
    spawnHealer() {
        if (this.wave === 10) {
            this.healers.push(new HealerPenance(new Position(36, 49), this.maxHealerHealth, this.ticks, this.currentHealerId));
        }
        else {
            this.healers.push(new HealerPenance(new Position(42, 37), this.maxHealerHealth, this.ticks, this.currentHealerId));
        }
        this.currentHealerId++;
        this.healersAlive++;
    }
    /**
     * Progresses each penance's state by one tick.
     *
     * @private
     */
    tickPenance() {
        this.healers.forEach((healer) => {
            healer.tick(this);
        });
        this.runners.forEach((runner) => {
            runner.tick(this);
        });
    }
    /**
     * Removes all to-be-removed Penance.
     *
     * @private
     */
    removePenance() {
        this.runnersToRemove.forEach((runnerToRemove) => {
            this.runners.splice(this.runners.indexOf(runnerToRemove), 1);
        });
        this.healersToRemove.forEach((healerToRemove) => {
            this.healers.splice(this.healers.indexOf(healerToRemove), 1);
        });
    }
    /**
     * Changes the defender food call to be one of the foods that it is currently not,
     * each with equal probability
     *
     * @private
     */
    changeDefenderFoodCall() {
        if (this.foodCallsIndex < this.foodCalls.length) {
            this.defenderFoodCall = this.foodCalls[this.foodCallsIndex];
            this.foodCallsIndex++;
            return;
        }
        switch (this.defenderFoodCall) {
            case FoodType.TOFU:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.CRACKERS;
                }
                else {
                    this.defenderFoodCall = FoodType.WORMS;
                }
                break;
            case FoodType.CRACKERS:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.WORMS;
                }
                else {
                    this.defenderFoodCall = FoodType.TOFU;
                }
                break;
            case FoodType.WORMS:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.TOFU;
                }
                else {
                    this.defenderFoodCall = FoodType.CRACKERS;
                }
                break;
            default:
                this.defenderFoodCall = FoodType.TOFU;
                break;
        }
    }
    /**
     * Determines if the tile with the given position blocks {@link Penance} movement
     * (i.e. Penance can not move onto the tile).
     *
     * @param position  the position of the tile to determine if Penance are blocked by
     * @return          true if the tile with the given position blocks Penance movement,
     *                  otherwise false
     */
    tileBlocksPenance(position) {
        if (position.equals(this.defenderPlayer.position)) {
            return true;
        }
        if (position.equals(this.collectorPlayer.position)) {
            return true;
        }
        if (position.y === 22) {
            if (position.x >= 20 && position.x <= 22) {
                return true;
            }
            if (this.wave !== 10 && position.x >= 39 && position.x <= 41) {
                return true;
            }
        }
        else if (position.x === 46 && position.y >= 9 && position.y <= 12) {
            return true;
        }
        else if (this.wave !== 10 && position.equals(new Position(27, 24))) {
            return true;
        }
        return false;
    }
    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    clone() {
        let barbarianAssault = new BarbarianAssault(this.wave, this.requireRepairs, this.requireLogs, this.infiniteFood, this.runnerMovements, this.defenderLevel, this.mainAttackerCommands, this.secondAttackerCommands, this.healerCommands, this.collectorCommands, this.defenderCommands, this.foodCalls);
        barbarianAssault.map = this.map === null ? null : this.map.clone();
        barbarianAssault.ticks = this.ticks;
        barbarianAssault.wave = this.wave;
        barbarianAssault.maxRunnersAlive = this.maxRunnersAlive;
        barbarianAssault.totalRunners = this.totalRunners;
        barbarianAssault.defenderFoodCall = this.defenderFoodCall;
        barbarianAssault.eastTrapCharges = this.eastTrapCharges;
        barbarianAssault.westTrapCharges = this.westTrapCharges;
        barbarianAssault.northwestLogsArePresent = this.northwestLogsArePresent;
        barbarianAssault.southeastLogsArePresent = this.southeastLogsArePresent;
        barbarianAssault.eastTrapPosition = this.eastTrapPosition === null ? null : this.eastTrapPosition.clone();
        barbarianAssault.westTrapPosition = this.westTrapPosition === null ? null : this.westTrapPosition.clone();
        barbarianAssault.runnersToRemove = [];
        for (let i = 0; i < this.runnersToRemove.length; i++) {
            barbarianAssault.runnersToRemove.push(this.runnersToRemove[i] === null ? null : this.runnersToRemove[i].clone());
        }
        barbarianAssault.runnersAlive = this.runnersAlive;
        barbarianAssault.runnersKilled = this.runnersKilled;
        barbarianAssault.collectorPlayer = this.collectorPlayer === null ? null : this.collectorPlayer.clone();
        barbarianAssault.defenderPlayer = this.defenderPlayer === null ? null : this.defenderPlayer.clone();
        barbarianAssault.mainAttackerPlayer = this.mainAttackerPlayer === null ? null : this.mainAttackerPlayer.clone();
        barbarianAssault.secondAttackerPlayer = this.secondAttackerPlayer === null ? null : this.secondAttackerPlayer.clone();
        barbarianAssault.healerPlayer = this.healerPlayer === null ? null : this.healerPlayer.clone();
        barbarianAssault.requireRepairs = this.requireRepairs;
        barbarianAssault.requireLogs = this.requireLogs;
        barbarianAssault.infiniteFood = this.infiniteFood;
        barbarianAssault.runners = [];
        for (let i = 0; i < this.runners.length; i++) {
            barbarianAssault.runners.push(this.runners[i] === null ? null : this.runners[i].clone());
        }
        barbarianAssault.runnerMovements = [...this.runnerMovements];
        barbarianAssault.runnerMovementsIndex = this.runnerMovementsIndex;
        barbarianAssault.currentRunnerId = this.currentRunnerId;
        barbarianAssault.defenderLevel = this.defenderLevel;
        barbarianAssault.mainAttackerCommands = new Map();
        this.mainAttackerCommands.forEach((position, tick) => {
            barbarianAssault.mainAttackerCommands.set(tick, position === null ? null : position.clone());
        });
        barbarianAssault.secondAttackerCommands = new Map();
        this.secondAttackerCommands.forEach((position, tick) => {
            barbarianAssault.secondAttackerCommands.set(tick, position === null ? null : position.clone());
        });
        barbarianAssault.healerCommands = new Map();
        this.healerCommands.forEach((position, tick) => {
            barbarianAssault.healerCommands.set(tick, position === null ? null : position.clone());
        });
        barbarianAssault.collectorCommands = new Map();
        this.collectorCommands.forEach((position, tick) => {
            barbarianAssault.collectorCommands.set(tick, position === null ? null : position.clone());
        });
        barbarianAssault.defenderCommands = new Map();
        this.defenderCommands.forEach((position, tick) => {
            barbarianAssault.defenderCommands.set(tick, position === null ? null : position.clone());
        });
        barbarianAssault.foodCalls = new Array;
        for (let i = 0; i < this.foodCalls.length; i++) {
            barbarianAssault.foodCalls.push(this.foodCalls[i]);
        }
        return barbarianAssault;
    }
}
