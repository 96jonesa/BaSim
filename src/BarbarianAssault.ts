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
import {Cannon} from "./Cannon.js";
import {CannonCommand} from "./CannonCommand.js";
import {HealerCodeCommand} from "./HealerCodeCommand.js";
import {HealerCodeAction} from "./HealerCodeAction.js";
import {Player} from "./Player.js";
import {WalkRunCommand} from "./WalkRunCommand.js";
import {ToggleRunCommand} from "./ToggleRunCommand.js";
import {SeedCommand} from "./SeedCommand.js";
import {SeedType} from "./SeedType.js";
import {RedXCommand} from "./RedXCommand.js";
import {RedXMoveCommand} from "./RedXMoveCommand.js";
import {DefenderPickupAtCommand} from "./DefenderPickupAtCommand.js";

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
    public ignoreMaxHealers: boolean = false;
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
    public cannon: Cannon = new Cannon();
    public healerSpawnTargets: Array<string> = [];
    public renderDistanceEnabled: boolean = false;
    public simpleFood: boolean = false;
    public seedQueuePath: boolean = true;
    public playerTickStartPositions: Map<Player, Position> = new Map();
    public runnerTickStartPositions: Map<number, Position> = new Map();
    public runnerSpawns: Array<number> = [];
    public runnerSpawnsIndex: number = 0;
    public healerSpawns: Array<number> = [];
    public healerSpawnsIndex: number = 0;


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
        foodCalls: Array<FoodType>,
        cannonQueue: Array<CannonCommand> = []
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
        this.cannon.queue = cannonQueue;

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
            this.defenderPlayer = new DefenderPlayer(new Position(28, 8));
            this.collectorPlayer = new CollectorPlayer(new Position(32, 8));
            this.mainAttackerPlayer = new AttackerPlayer(new Position(30, 10));
            this.secondAttackerPlayer = new AttackerPlayer(new Position(29, 9));
            this.healerPlayer = new HealerPlayer(new Position(31, 9));
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
        this.healersToRemove.length = 0;

        this.playerTickStartPositions.set(this.mainAttackerPlayer, this.mainAttackerPlayer.position.clone());
        this.playerTickStartPositions.set(this.secondAttackerPlayer, this.secondAttackerPlayer.position.clone());
        this.playerTickStartPositions.set(this.healerPlayer, this.healerPlayer.position.clone());
        this.playerTickStartPositions.set(this.collectorPlayer, this.collectorPlayer.position.clone());
        this.playerTickStartPositions.set(this.defenderPlayer, this.defenderPlayer.position.clone());

        this.runnerTickStartPositions.clear();
        for (const runner of this.runners) {
            if (runner !== null) {
                this.runnerTickStartPositions.set(runner.id, runner.position.clone());
            }
        }

        this.applySeedMovements();
        this.checkPhasing(this.playerTickStartPositions, (p) => p.seedMovedThisTick);

        this.tickPenance();
        this.removePenance();
        this.cannon.tick(this);

        if (this.ticks > 1 && this.ticks % 10 === 1) {
            this.northwestLogsArePresent = true;
            this.southeastLogsArePresent = true;
        }

        if (this.ticks > 2 && this.ticks % 50 === 2 && !this.simpleFood) {
            this.changeDefenderFoodCall();
        }

        const isDefaultCycle: boolean = this.ticks > 1 && this.ticks % 10 === 1;
        const shouldSpawnRunner: boolean = this.runnerSpawns.length === 0
            ? isDefaultCycle
            : this.runnerSpawnsIndex < this.runnerSpawns.length && this.runnerSpawns[this.runnerSpawnsIndex] === this.ticks;
        if (shouldSpawnRunner && this.runnersAlive < this.maxRunnersAlive && this.runnersKilled + this.runnersAlive < this.totalRunners) {
            this.spawnRunner();
            if (this.runnerSpawns.length > 0) {
                this.runnerSpawnsIndex++;
            }
        }

        const shouldSpawnHealer: boolean = this.healerSpawns.length === 0
            ? isDefaultCycle
            : this.healerSpawnsIndex < this.healerSpawns.length && this.healerSpawns[this.healerSpawnsIndex] === this.ticks;
        if (shouldSpawnHealer && (this.ignoreMaxHealers || this.healersAlive < this.maxHealersAlive) && this.healersKilled + this.healersAlive < this.totalHealers) {
            this.spawnHealer();
            if (this.healerSpawns.length > 0) {
                this.healerSpawnsIndex++;
            }
        }

        const prePathPositions: Map<Player, Position> = new Map();
        for (const player of this.allPlayers()) {
            prePathPositions.set(player, player.position.clone());
        }

        this.tickPlayers();
        this.checkPathPhasing(prePathPositions);

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

    private allPlayers(): Array<Player> {
        return [this.mainAttackerPlayer, this.secondAttackerPlayer, this.healerPlayer, this.collectorPlayer, this.defenderPlayer];
    }

    /**
     * Checks for player phasing after the path movement window. A player becomes phased if
     * another player moved onto and then off their tile (including intermediate steps),
     * or was on the same tile before the window and moved off during it,
     * unless the player also moved during the window.
     */
    private checkPathPhasing(prePositions: Map<Player, Position>): void {
        const players = this.allPlayers();
        for (const player of players) {
            const didMove = !player.position.equals(prePositions.get(player));
            if (didMove) {
                player.phased = false;
                continue;
            }
            const playerPos = prePositions.get(player);
            for (const other of players) {
                if (other === player) continue;
                const otherStartedHere = prePositions.get(other).equals(playerPos);
                const otherSteppedHere = other.pathStepPositions.some(p => p.equals(playerPos));
                const otherEndedHere = other.position.equals(playerPos);
                if ((otherStartedHere || otherSteppedHere) && !otherEndedHere) {
                    player.phased = true;
                    break;
                }
            }
        }
    }

    /**
     * Checks for player phasing after a movement window. A player becomes phased if
     * another player was on the same tile before the window and moved off during it,
     * unless the player also moved during the window.
     */
    private checkPhasing(prePositions: Map<Player, Position>, didMove: (p: Player) => boolean): void {
        const players = this.allPlayers();
        for (const player of players) {
            if (didMove(player)) {
                player.phased = false;
                continue;
            }
            const prePosP = prePositions.get(player);
            for (const other of players) {
                if (other === player) continue;
                const prePosQ = prePositions.get(other);
                if (prePosQ.equals(prePosP) && didMove(other)) {
                    player.phased = true;
                    break;
                }
            }
        }
    }

    private applySeedMovements(): void {
        this.applySeedForPlayer(this.mainAttackerPlayer);
        this.applySeedForPlayer(this.secondAttackerPlayer);
        this.applySeedForPlayer(this.healerPlayer);
        this.applySeedForPlayer(this.collectorPlayer);
        this.applySeedForPlayer(this.defenderPlayer);
    }

    private applySeedForPlayer(player: Player): void {
        player.seedMovedThisTick = false;

        // Can't use a seed while repairing
        if (player instanceof DefenderPlayer && player.repairTicksRemaining > 0) {
            player.pendingSeed = null;
            player.repeatSeedType = null;
            player.preSeedPosition = null;
            player.seedMovedToPosition = null;
            return;
        }

        // Check for auto-repeat from previous tick
        if (player.repeatSeedType !== null) {
            const seedType = player.repeatSeedType;
            const preSeedPos = player.preSeedPosition;
            const blockedTile = player.seedMovedToPosition;

            player.repeatSeedType = null;
            player.preSeedPosition = null;
            player.seedMovedToPosition = null;

            // Can't use a seed two ticks in a row
            player.pendingSeed = null;

            // If player returned to pre-seed position, apply auto-repeat
            if (preSeedPos !== null && player.position.equals(preSeedPos)) {
                player.seedMovedThisTick = true;
                this.applySeedStep(player, seedType, blockedTile);

                if (player.checkpointIndex < player.checkpoints.length &&
                    player.position.equals(player.checkpoints[player.checkpointIndex])) {
                    player.checkpointIndex++;
                }
                // Do NOT recalculate path
            }

            return;
        }

        // Handle regular pending seed
        if (player.pendingSeed === null) return;

        const seedType = player.pendingSeed;
        player.pendingSeed = null;
        player.seedMovedThisTick = true;

        // Save state for repeat detection
        player.preSeedPosition = player.position.clone();

        this.applySeedStep(player, seedType, null);

        // Save where seed moved player to (blocked on repeat)
        player.seedMovedToPosition = player.position.clone();
        player.repeatSeedType = seedType;

        if (player.checkpointIndex < player.checkpoints.length &&
            player.position.equals(player.checkpoints[player.checkpointIndex])) {
            player.checkpointIndex++;
        }

        if (player.pathDestination !== null) {
            player.findPath(this, player.pathDestination);
        }
    }

    private applySeedStep(player: Player, seedType: SeedType, blockedTile: Position): void {
        const directions = seedType === "MITHRIL"
            ? [[-1, 0], [1, 0], [0, -1], [0, 1]]   // W, E, S, N
            : [[1, 0], [-1, 0], [0, -1], [0, 1]];   // E, W, S, N

        for (const [dx, dy] of directions) {
            const targetX = player.position.x + dx;
            const targetY = player.position.y + dy;

            if (blockedTile !== null && targetX === blockedTile.x && targetY === blockedTile.y) {
                continue;
            }

            const canMove = dx === -1 ? this.map.canMoveWest(player.position)
                : dx === 1 ? this.map.canMoveEast(player.position)
                : dy === -1 ? this.map.canMoveSouth(player.position)
                : this.map.canMoveNorth(player.position);

            if (canMove) {
                player.position.x += dx;
                player.position.y += dy;
                return;
            }
        }
    }

    /**
     * Executes player commands for all players for the current tick.
     *
     * @private
     */
    private executePlayerCommands(): void {
        if (this.mainAttackerCommands.has(this.ticks)) {
            const commands = this.mainAttackerCommands.get(this.ticks);
            const hasMoveCommand = commands.some(c => c instanceof MoveCommand);
            let seedCommandProcessed = false;
            commands.forEach((command: Command): void => {
                if (this.mainAttackerPlayer.seedMovedThisTick && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (seedCommandProcessed && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (command instanceof MoveCommand) {
                    this.mainAttackerPlayer.clearCodeQueue();
                    this.mainAttackerPlayer.isRedXPath = false;
                    this.mainAttackerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXMoveCommand) {
                    this.mainAttackerPlayer.clearCodeQueue();
                    this.mainAttackerPlayer.isRedXPath = true;
                    this.mainAttackerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXCommand) {
                    this.mainAttackerPlayer.redXHealerId = command.healerId;
                } else if (command instanceof HealerCodeCommand) {
                    this.expandHealerCodeCommand(command, this.mainAttackerPlayer);
                    this.mainAttackerPlayer.initializeFoodPath(this);
                } else if (command instanceof WalkRunCommand) {
                    this.mainAttackerPlayer.isRunning = command.isRunning;
                } else if (command instanceof ToggleRunCommand) {
                    this.mainAttackerPlayer.isRunning = !this.mainAttackerPlayer.isRunning;
                } else if (command instanceof SeedCommand) {
                    this.mainAttackerPlayer.pendingSeed = command.seedType;
                    this.mainAttackerPlayer.clearCodeQueue();
                    if (!this.seedQueuePath && !hasMoveCommand) {
                        this.mainAttackerPlayer.pathDestination = null;
                        this.mainAttackerPlayer.isRedXPath = false;
                        this.mainAttackerPlayer.checkpoints = [];
                        this.mainAttackerPlayer.checkpointIndex = 0;
                    }
                    seedCommandProcessed = true;
                }
            });
        }

        if (this.secondAttackerCommands.has(this.ticks)) {
            const commands = this.secondAttackerCommands.get(this.ticks);
            const hasMoveCommand = commands.some(c => c instanceof MoveCommand);
            let seedCommandProcessed = false;
            commands.forEach((command: Command): void => {
                if (this.secondAttackerPlayer.seedMovedThisTick && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (seedCommandProcessed && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (command instanceof MoveCommand) {
                    this.secondAttackerPlayer.clearCodeQueue();
                    this.secondAttackerPlayer.isRedXPath = false;
                    this.secondAttackerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXMoveCommand) {
                    this.secondAttackerPlayer.clearCodeQueue();
                    this.secondAttackerPlayer.isRedXPath = true;
                    this.secondAttackerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXCommand) {
                    this.secondAttackerPlayer.redXHealerId = command.healerId;
                } else if (command instanceof HealerCodeCommand) {
                    this.expandHealerCodeCommand(command, this.secondAttackerPlayer);
                    this.secondAttackerPlayer.initializeFoodPath(this);
                } else if (command instanceof WalkRunCommand) {
                    this.secondAttackerPlayer.isRunning = command.isRunning;
                } else if (command instanceof ToggleRunCommand) {
                    this.secondAttackerPlayer.isRunning = !this.secondAttackerPlayer.isRunning;
                } else if (command instanceof SeedCommand) {
                    this.secondAttackerPlayer.pendingSeed = command.seedType;
                    this.secondAttackerPlayer.clearCodeQueue();
                    if (!this.seedQueuePath && !hasMoveCommand) {
                        this.secondAttackerPlayer.pathDestination = null;
                        this.secondAttackerPlayer.isRedXPath = false;
                        this.secondAttackerPlayer.checkpoints = [];
                        this.secondAttackerPlayer.checkpointIndex = 0;
                    }
                    seedCommandProcessed = true;
                }
            });
        }

        if (this.healerCommands.has(this.ticks)) {
            const commands = this.healerCommands.get(this.ticks);
            const hasMoveCommand = commands.some(c => c instanceof MoveCommand);
            let seedCommandProcessed = false;
            commands.forEach((command: Command): void => {
                if (this.healerPlayer.seedMovedThisTick && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (seedCommandProcessed && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (command instanceof MoveCommand) {
                    this.healerPlayer.clearCodeQueue();
                    this.healerPlayer.isRedXPath = false;
                    this.healerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXMoveCommand) {
                    this.healerPlayer.clearCodeQueue();
                    this.healerPlayer.isRedXPath = true;
                    this.healerPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXCommand) {
                    this.healerPlayer.redXHealerId = command.healerId;
                } else if (command instanceof HealerCodeCommand) {
                    this.expandHealerCodeCommand(command, this.healerPlayer);
                    this.healerPlayer.initializeFoodPath(this);
                } else if (command instanceof WalkRunCommand) {
                    this.healerPlayer.isRunning = command.isRunning;
                } else if (command instanceof ToggleRunCommand) {
                    this.healerPlayer.isRunning = !this.healerPlayer.isRunning;
                } else if (command instanceof SeedCommand) {
                    this.healerPlayer.pendingSeed = command.seedType;
                    this.healerPlayer.clearCodeQueue();
                    if (!this.seedQueuePath && !hasMoveCommand) {
                        this.healerPlayer.pathDestination = null;
                        this.healerPlayer.isRedXPath = false;
                        this.healerPlayer.checkpoints = [];
                        this.healerPlayer.checkpointIndex = 0;
                    }
                    seedCommandProcessed = true;
                }
            });
        }

        if (this.collectorCommands.has(this.ticks)) {
            const commands = this.collectorCommands.get(this.ticks);
            const hasMoveCommand = commands.some(c => c instanceof MoveCommand);
            let seedCommandProcessed = false;
            commands.forEach((command: Command): void => {
                if (this.collectorPlayer.seedMovedThisTick && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (seedCommandProcessed && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (command instanceof MoveCommand) {
                    this.collectorPlayer.clearCodeQueue();
                    this.collectorPlayer.isRedXPath = false;
                    this.collectorPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXMoveCommand) {
                    this.collectorPlayer.clearCodeQueue();
                    this.collectorPlayer.isRedXPath = true;
                    this.collectorPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXCommand) {
                    this.collectorPlayer.redXHealerId = command.healerId;
                } else if (command instanceof HealerCodeCommand) {
                    this.expandHealerCodeCommand(command, this.collectorPlayer);
                    this.collectorPlayer.initializeFoodPath(this);
                } else if (command instanceof WalkRunCommand) {
                    this.collectorPlayer.isRunning = command.isRunning;
                } else if (command instanceof ToggleRunCommand) {
                    this.collectorPlayer.isRunning = !this.collectorPlayer.isRunning;
                } else if (command instanceof SeedCommand) {
                    this.collectorPlayer.pendingSeed = command.seedType;
                    this.collectorPlayer.clearCodeQueue();
                    if (!this.seedQueuePath && !hasMoveCommand) {
                        this.collectorPlayer.pathDestination = null;
                        this.collectorPlayer.isRedXPath = false;
                        this.collectorPlayer.checkpoints = [];
                        this.collectorPlayer.checkpointIndex = 0;
                    }
                    seedCommandProcessed = true;
                }
            });
        }

        if (this.defenderCommands.has(this.ticks)) {
            const commands = this.defenderCommands.get(this.ticks);
            const hasMoveCommand = commands.some(c => c instanceof MoveCommand);
            let seedCommandProcessed = false;
            commands.forEach((command: Command): void => {
                if (this.defenderPlayer.seedMovedThisTick && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (seedCommandProcessed && !(command instanceof WalkRunCommand) && !(command instanceof ToggleRunCommand)) return;
                if (command instanceof MoveCommand) {
                    this.defenderPlayer.clearCodeQueue();
                    this.defenderPlayer.isRedXPath = false;
                    this.defenderPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXMoveCommand) {
                    this.defenderPlayer.clearCodeQueue();
                    this.defenderPlayer.isRedXPath = true;
                    this.defenderPlayer.findPath(this, command.destination.clone());
                } else if (command instanceof RedXCommand) {
                    this.defenderPlayer.redXHealerId = command.healerId;
                } else if (command instanceof HealerCodeCommand) {
                    this.expandHealerCodeCommand(command, this.defenderPlayer);
                    this.defenderPlayer.initializeFoodPath(this);
                } else if (command instanceof WalkRunCommand) {
                    this.defenderPlayer.isRunning = command.isRunning;
                } else if (command instanceof ToggleRunCommand) {
                    this.defenderPlayer.isRunning = !this.defenderPlayer.isRunning;
                } else if (command instanceof SeedCommand) {
                    this.defenderPlayer.pendingSeed = command.seedType;
                    this.defenderPlayer.clearCodeQueue();
                    if (!this.seedQueuePath && !hasMoveCommand) {
                        this.defenderPlayer.pathDestination = null;
                        this.defenderPlayer.isRedXPath = false;
                        this.defenderPlayer.checkpoints = [];
                        this.defenderPlayer.checkpointIndex = 0;
                    }
                    seedCommandProcessed = true;
                } else if (command instanceof DefenderActionCommand) {
                    this.defenderPlayer.clearCodeQueue();
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
                        case DefenderActionType.PICKUP_ANY_FOOD:
                            this.defenderPlayer.shouldPickUpAnyFood = true;
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
                } else if (command instanceof DefenderPickupAtCommand) {
                    this.defenderPlayer.pickUpFoodAtPosition = command.position.clone();
                }
            });
        }
    }

    private expandHealerCodeCommand(command: HealerCodeCommand, player: Player): void {
        player.clearCodeQueue();
        for (let i: number = 0; i < command.count; i++) {
            player.codeQueue.push(new HealerCodeAction(command.healerId, 0));
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
        const spawnPos = this.wave === 10 ? new Position(36, 39) : new Position(42, 37);
        const healer = new HealerPenance(spawnPos, this.maxHealerHealth, this.ticks, this.currentHealerId);

        const spawnIndex = this.currentHealerId - 1;
        if (spawnIndex < this.healerSpawnTargets.length) {
            const targetStr = this.healerSpawnTargets[spawnIndex];
            let forced = "";
            if (targetStr.includes("m")) forced += "main";
            if (targetStr.includes("2")) forced += "second";
            if (targetStr.includes("h")) forced += "heal";
            if (targetStr.includes("c")) forced += "collector";
            if (targetStr.includes("d")) forced += "player";
            healer.forcedTarget = forced;
        }

        this.healers.push(healer);
        this.currentHealerId++;
        this.healersAlive++;
    }

    /**
     * Progresses each penance's state by one tick.
     *
     * @private
     */
    private tickPenance(): void {
        this.runners.forEach((runner: RunnerPenance): void => {
            runner.tick(this);
        });

        this.healers.forEach((healer: HealerPenance): void => {
            healer.tick(this);
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
    public isHealerRedXBlocked(healer: HealerPenance): boolean {
        const players = [this.mainAttackerPlayer, this.secondAttackerPlayer, this.healerPlayer, this.collectorPlayer, this.defenderPlayer];
        for (const player of players) {
            if (player.redXHealerId === healer.id && player.isRedXPath && player.position.equals(healer.position)) {
                return true;
            }
        }
        return false;
    }

    public tileBlocksPenance(position: Position): boolean {
        if (!this.defenderPlayer.phased && position.equals(this.defenderPlayer.position)) {
            return true;
        }

        if (!this.collectorPlayer.phased && position.equals(this.collectorPlayer.position)) {
            return true;
        }

        if (!this.mainAttackerPlayer.phased && position.equals(this.mainAttackerPlayer.position)) {
            return true;
        }

        if (!this.secondAttackerPlayer.phased && position.equals(this.secondAttackerPlayer.position)) {
            return true;
        }

        if (!this.healerPlayer.phased && position.equals(this.healerPlayer.position)) {
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
        barbarianAssault.healersToRemove = [];
        for (let i: number = 0; i < this.healersToRemove.length; i++) {
            barbarianAssault.healersToRemove.push(this.healersToRemove[i] === null ? null : this.healersToRemove[i].clone());
        }
        barbarianAssault.healers = [];
        for (let i: number = 0; i < this.healers.length; i++) {
            barbarianAssault.healers.push(this.healers[i] === null ? null : this.healers[i].clone());
        }
        barbarianAssault.healersAlive = this.healersAlive;
        barbarianAssault.healersKilled = this.healersKilled;
        barbarianAssault.currentHealerId = this.currentHealerId;
        barbarianAssault.foodCallsIndex = this.foodCallsIndex;
        barbarianAssault.collectorPlayer = this.collectorPlayer === null ? null : this.collectorPlayer.clone();
        barbarianAssault.defenderPlayer = this.defenderPlayer === null ? null : this.defenderPlayer.clone();
        barbarianAssault.mainAttackerPlayer = this.mainAttackerPlayer === null ? null : this.mainAttackerPlayer.clone();
        barbarianAssault.secondAttackerPlayer = this.secondAttackerPlayer === null ? null : this.secondAttackerPlayer.clone();
        barbarianAssault.healerPlayer = this.healerPlayer === null ? null : this.healerPlayer.clone();
        barbarianAssault.requireRepairs = this.requireRepairs;
        barbarianAssault.requireLogs = this.requireLogs;
        barbarianAssault.infiniteFood = this.infiniteFood;
        barbarianAssault.ignoreMaxHealers = this.ignoreMaxHealers;
        barbarianAssault.runners = [];
        for (let i: number = 0; i < this.runners.length; i++) {
            barbarianAssault.runners.push(this.runners[i] === null ? null : this.runners[i].clone());
        }
        // Re-link runner foodTargets to the cloned map's food objects
        for (const runner of barbarianAssault.runners) {
            if (runner !== null && runner.foodTarget !== null) {
                const ft = runner.foodTarget;
                const foodZone = barbarianAssault.map.getFoodZone(ft.position.x >>> 3, ft.position.y >>> 3);
                for (const food of foodZone.foodList) {
                    if (food.position.x === ft.position.x && food.position.y === ft.position.y && food.type === ft.type && food.isGood === ft.isGood) {
                        runner.foodTarget = food;
                        break;
                    }
                }
            }
        }
        // Re-link healer targets to cloned players/runners
        for (let i = 0; i < this.healers.length; i++) {
            const originalHealer = this.healers[i];
            const clonedHealer = barbarianAssault.healers[i];
            if (originalHealer === null || clonedHealer === null || originalHealer.target === null) continue;

            if (originalHealer.target === this.mainAttackerPlayer) {
                clonedHealer.target = barbarianAssault.mainAttackerPlayer;
            } else if (originalHealer.target === this.secondAttackerPlayer) {
                clonedHealer.target = barbarianAssault.secondAttackerPlayer;
            } else if (originalHealer.target === this.healerPlayer) {
                clonedHealer.target = barbarianAssault.healerPlayer;
            } else if (originalHealer.target === this.collectorPlayer) {
                clonedHealer.target = barbarianAssault.collectorPlayer;
            } else if (originalHealer.target === this.defenderPlayer) {
                clonedHealer.target = barbarianAssault.defenderPlayer;
            } else if (originalHealer.target instanceof RunnerPenance) {
                const targetId = (originalHealer.target as RunnerPenance).id;
                clonedHealer.target = barbarianAssault.runners.find(r => r !== null && r.id === targetId) || null;
            }
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
        barbarianAssault.cannon = this.cannon.clone();
        barbarianAssault.healerSpawnTargets = [...this.healerSpawnTargets];
        barbarianAssault.renderDistanceEnabled = this.renderDistanceEnabled;
        barbarianAssault.simpleFood = this.simpleFood;
        barbarianAssault.seedQueuePath = this.seedQueuePath;
        barbarianAssault.playerTickStartPositions = new Map();
        const playerMapping: Array<[Player, Player]> = [
            [this.mainAttackerPlayer, barbarianAssault.mainAttackerPlayer],
            [this.secondAttackerPlayer, barbarianAssault.secondAttackerPlayer],
            [this.healerPlayer, barbarianAssault.healerPlayer],
            [this.collectorPlayer, barbarianAssault.collectorPlayer],
            [this.defenderPlayer, barbarianAssault.defenderPlayer],
        ];
        for (const [original, cloned] of playerMapping) {
            const pos = this.playerTickStartPositions.get(original);
            if (pos) {
                barbarianAssault.playerTickStartPositions.set(cloned, pos.clone());
            }
        }
        barbarianAssault.runnerTickStartPositions = new Map();
        for (const [id, pos] of this.runnerTickStartPositions) {
            barbarianAssault.runnerTickStartPositions.set(id, pos.clone());
        }
        barbarianAssault.runnerSpawns = [...this.runnerSpawns];
        barbarianAssault.runnerSpawnsIndex = this.runnerSpawnsIndex;
        barbarianAssault.healerSpawns = [...this.healerSpawns];
        barbarianAssault.healerSpawnsIndex = this.healerSpawnsIndex;

        return barbarianAssault;
    }
}