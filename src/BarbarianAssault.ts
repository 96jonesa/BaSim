import {FoodType} from "./FoodType.js";
import {Position} from "./Position.js";
import {BarbarianAssaultMap} from "./BarbarianAssaultMap.js";
import {RunnerPenance} from "./RunnerPenance.js";
import {DefenderPlayer} from "./DefenderPlayer.js";
import {RunnerPenanceRng} from "./RunnerPenanceRng.js";
import {CollectorPlayer} from "./CollectorPlayer.js";
import {AttackerPlayer} from "./AttackerPlayer.js";
import {HealerPlayer} from "./HealerPlayer.js";
import {HealerPenance} from "./HealerPenance.js";
import {Command} from "./Command.js";
import {MoveCommand} from "./MoveCommand.js";
import {DefenderActionCommand} from "./DefenderActionCommand.js";
import {DefenderActionType} from "./DefenderActionType.js";

/**
 * Represents a game of Barbarian Assault: holds state information and exposes functions for
 * progressing the game state.
 */
export class BarbarianAssault {
    public map: BarbarianAssaultMap;
    public ticks: number = 0;
    public wave: number;
    public maxRunnersAlive: number;
    public totalRunners: number;
    public defenderFoodCall: FoodType;
    public eastTrapCharges: number = 2;
    public westTrapCharges: number = 2;
    public northwestLogsArePresent: boolean = true;
    public southeastLogsArePresent: boolean = true;
    public eastTrapPosition: Position = new Position(45, 26);
    public westTrapPosition: Position = new Position(15, 25);
    public northwestLogsPosition: Position;
    public southeastLogsPosition: Position;
    public runnersToRemove: Array<RunnerPenance> = [];
    public runnersAlive: number = 0;
    public runnersKilled: number = 0;
    public healersToRemove: Array<HealerPenance> = [];
    public healers: Array<HealerPenance> = [];
    public healersAlive: number = 0;
    public healersKilled: number = 0;
    public maxHealersAlive: number;
    public totalHealers: number;
    public maxHealerHealth: number;
    public collectorPlayer: CollectorPlayer;
    public defenderPlayer: DefenderPlayer;
    public mainAttackerPlayer: AttackerPlayer;
    public secondAttackerPlayer: AttackerPlayer;
    public healerPlayer: HealerPlayer;
    public requireRepairs: boolean;
    public requireLogs: boolean;
    public infiniteFood: boolean;
    public runners: Array<RunnerPenance> = [];
    public runnerMovements: Array<string>;
    public runnerMovementsIndex: number = 0;
    public currentRunnerId: number = 1;
    public currentHealerId: number = 1;
    public defenderLevel: number;
    public mainAttackerCommands: Map<number, Array<Command>>;
    public secondAttackerCommands: Map<number, Array<Command>>;
    public healerCommands: Map<number, Array<Command>>;
    public collectorCommands: Map<number, Array<Command>>;
    public defenderCommands: Map<number, Array<Command>>;
    public foodCalls: Array<FoodType>;
    public foodCallsIndex: number = 0;


    public constructor(
        wave: number,
        requireRepairs: boolean,
        requireLogs: boolean,
        infiniteFood: boolean,
        runnerMovements: Array<string>,
        defenderLevel: number,
        mainAttackerCommands: Map<number, Array<Command>>,
        secondAttackerCommands: Map<number, Array<Command>>,
        healerCommands: Map<number, Array<Command>>,
        collectorCommands: Map<number, Array<Command>>,
        defenderCommands: Map<number, Array<Command>>,
        foodCalls: Array<FoodType>
    ) {
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
        } else {
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
    public tick(): void {
        this.ticks++;
        // console.log(this.ticks);
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
        this.executePlayerCommands();
    }

    /**
     * Progresses each player's state by one tick.
     *
     * @private
     */
    private tickPlayers(): void {
        this.defenderPlayer.tick(this);
        this.collectorPlayer.tick(this);
        this.mainAttackerPlayer.tick(this);
        this.secondAttackerPlayer.tick(this);
        this.healerPlayer.tick(this);
    }

    /**
     * Executes player commands for all players for the current tick.
     *
     * @private
     */
    private executePlayerCommands(): void {
        if (this.mainAttackerCommands.has(this.ticks)) {
            this.mainAttackerCommands.get(this.ticks).forEach((command: Command): void => {
                if (command instanceof MoveCommand) {
                    this.mainAttackerPlayer.findPath(this, command.destination.clone());
                }
            });
        }

        if (this.secondAttackerCommands.has(this.ticks)) {
            this.secondAttackerCommands.get(this.ticks).forEach((command: Command): void => {
                if (command instanceof MoveCommand) {
                    this.secondAttackerPlayer.findPath(this, command.destination.clone());
                }
            });
        }

        if (this.healerCommands.has(this.ticks)) {
            this.healerCommands.get(this.ticks).forEach((command: Command): void => {
                if (command instanceof MoveCommand) {
                    this.healerPlayer.findPath(this, command.destination.clone());
                }
            });
        }

        if (this.collectorCommands.has(this.ticks)) {
            this.collectorCommands.get(this.ticks).forEach((command: Command): void => {
                if (command instanceof MoveCommand) {
                    this.collectorPlayer.findPath(this, command.destination.clone());
                }
            });
        }

        if (this.defenderCommands.has(this.ticks)) {
            this.defenderCommands.get(this.ticks).forEach((command: Command): void => {
                if (command instanceof MoveCommand) {
                    this.defenderPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof DefenderActionCommand) {
                    switch (command.type) {
                        case DefenderActionType.DROP_TOFU:
                            this.defenderPlayer.dropFood(this, FoodType.TOFU);
                            break;
                        case DefenderActionType.DROP_CRACKERS:
                            this.defenderPlayer.dropFood(this, FoodType.CRACKERS);
                            break;
                        case DefenderActionType.DROP_WORMS:
                            this.defenderPlayer.dropFood(this, FoodType.WORMS);
                            break;
                        case DefenderActionType.PICKUP_TOFU:
                            this.defenderPlayer.foodBeingPickedUp = FoodType.TOFU;
                            break;
                        case DefenderActionType.PICKUP_CRACKERS:
                            this.defenderPlayer.foodBeingPickedUp = FoodType.CRACKERS;
                            break;
                        case DefenderActionType.PICKUP_WORMS:
                            this.defenderPlayer.foodBeingPickedUp = FoodType.WORMS;
                            break;
                        case DefenderActionType.PICKUP_LOGS:
                            this.defenderPlayer.isPickingUpLogs = true;
                            break;
                        case DefenderActionType.REPAIR_TRAP:
                            this.defenderPlayer.startRepairing(this);
                            break;
                        default:
                            break;
                    }
                }
            });
        }
    }

    /**
     * Spawns a RunnerPenance.
     *
     * @private
     */
    private spawnRunner(): void {
        let movements: string;

        if (this.runnerMovements.length > this.runnerMovementsIndex) {
            movements = this.runnerMovements[this.runnerMovementsIndex];
            this.runnerMovementsIndex++;
        } else {
            movements = "";
        }

        if (this.wave === 10) {
            this.runners.push(new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(movements), this.currentRunnerId, this.defenderLevel < 2 ? 4 : 5));
        } else {
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
    private spawnHealer(): void {
        if (this.wave === 10) {
            this.healers.push(new HealerPenance(new Position(36, 49), this.maxHealerHealth, this.ticks, this.currentHealerId));
        } else {
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
    private tickPenance(): void {
        this.healers.forEach((healer: HealerPenance): void => {
            healer.tick(this);
        });

        this.runners.forEach((runner: RunnerPenance): void => {
            runner.tick(this);
        });
    }

    /**
     * Removes all to-be-removed Penance.
     *
     * @private
     */
    private removePenance(): void {
        this.runnersToRemove.forEach((runnerToRemove: RunnerPenance): void => {
            this.runners.splice(this.runners.indexOf(runnerToRemove), 1);
        });

        this.healersToRemove.forEach((healerToRemove: HealerPenance): void => {
            this.healers.splice(this.healers.indexOf(healerToRemove), 1);
        });
    }

    /**
     * Changes the defender food call to be one of the foods that it is currently not,
     * each with equal probability
     *
     * @private
     */
    private changeDefenderFoodCall(): void {
        if (this.foodCallsIndex < this.foodCalls.length) {
            this.defenderFoodCall = this.foodCalls[this.foodCallsIndex];
            this.foodCallsIndex++;

            return;
        }

        switch (this.defenderFoodCall) {
            case FoodType.TOFU:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.CRACKERS;
                } else {
                    this.defenderFoodCall = FoodType.WORMS;
                }

                break;
            case FoodType.CRACKERS:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.WORMS;
                } else {
                    this.defenderFoodCall = FoodType.TOFU;
                }

                break;
            case FoodType.WORMS:
                if (Math.random() < 0.5) {
                    this.defenderFoodCall = FoodType.TOFU;
                } else {
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
    public tileBlocksPenance(position: Position): boolean {
        if (position.equals(this.defenderPlayer.position)) {
            return true;
        }

        if (position.equals(this.collectorPlayer.position)) {
            return true;
        }

        if (position.equals(this.mainAttackerPlayer.position)) {
            return true;
        }

        if (position.equals(this.secondAttackerPlayer.position)) {
            return true;
        }

        if (position.equals(this.healerPlayer.position)) {
            return true;
        }

        if (position.y === 22) {
            if (position.x >= 20 && position.x <= 22) {
                return true;
            }

            if (this.wave !== 10 && position.x >= 39 && position.x <= 41) {
                return true;
            }
        } else if (position.x === 46 && position.y >= 9 && position.y <= 12) {
            return true;
        } else if (this.wave !== 10 && position.equals(new Position(27, 24))) {
            return true;
        }

        return false;
    }

    /**
     * Creates a deep clone of this object.
     *
     * @return  a deep clone of this object
     */
    public clone(): BarbarianAssault {
        let barbarianAssault: BarbarianAssault = new BarbarianAssault(
            this.wave,
            this.requireRepairs,
            this.requireLogs,
            this.infiniteFood,
            this.runnerMovements,
            this.defenderLevel,
            this.mainAttackerCommands,
            this.secondAttackerCommands,
            this.healerCommands,
            this.collectorCommands,
            this.defenderCommands,
            this.foodCalls
        );
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
        for (let i: number = 0; i < this.runnersToRemove.length; i++) {
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
        for (let i: number = 0; i < this.runners.length; i++) {
            barbarianAssault.runners.push(this.runners[i] === null ? null : this.runners[i].clone());
        }
        barbarianAssault.runnerMovements = [...this.runnerMovements];
        barbarianAssault.runnerMovementsIndex = this.runnerMovementsIndex;
        barbarianAssault.currentRunnerId = this.currentRunnerId;
        barbarianAssault.defenderLevel = this.defenderLevel;
        barbarianAssault.mainAttackerCommands = new Map<number, Array<Command>>();
        this.mainAttackerCommands.forEach((commands: Array<Command>, tick: number): void => {
            const commandsArray: Array<Command> = [];

            commands.forEach((command: Command): void => {
                commandsArray.push(command.clone());
            });

            barbarianAssault.mainAttackerCommands.set(tick, commandsArray);
        });
        this.secondAttackerCommands.forEach((commands: Array<Command>, tick: number): void => {
            const commandsArray: Array<Command> = [];

            commands.forEach((command: Command): void => {
                commandsArray.push(command.clone());
            });

            barbarianAssault.secondAttackerCommands.set(tick, commandsArray);
        });
        this.healerCommands.forEach((commands: Array<Command>, tick: number): void => {
            const commandsArray: Array<Command> = [];

            commands.forEach((command: Command): void => {
                commandsArray.push(command.clone());
            });

            barbarianAssault.healerCommands.set(tick, commandsArray);
        });
        this.collectorCommands.forEach((commands: Array<Command>, tick: number): void => {
            const commandsArray: Array<Command> = [];

            commands.forEach((command: Command): void => {
                commandsArray.push(command.clone());
            });

            barbarianAssault.collectorCommands.set(tick, commandsArray);
        });
        this.defenderCommands.forEach((commands: Array<Command>, tick: number): void => {
            const commandsArray: Array<Command> = [];

            commands.forEach((command: Command): void => {
                commandsArray.push(command.clone());
            });

            barbarianAssault.defenderCommands.set(tick, commandsArray);
        });
        barbarianAssault.foodCalls = new Array<FoodType>;
        for (let i: number = 0; i < this.foodCalls.length; i++) {
            barbarianAssault.foodCalls.push(this.foodCalls[i]);
        }

        return barbarianAssault;
    }
}