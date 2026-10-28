import {describe, expect, test} from "vitest";
import {HealerPlayer} from "../src/HealerPlayer.js";
import {Position} from "../src/Position.js";
import {BarbarianAssault} from "../src/BarbarianAssault.js";
import {Command} from "../src/Command.js";
import {HealerCodeAction} from "../src/HealerCodeAction.js";
import {HealerPenance} from "../src/HealerPenance.js";
import {MoveCommand} from "../src/MoveCommand.js";
import {HealerCodeCommand} from "../src/HealerCodeCommand.js";

describe("tick", (): void => {
    test("does nothing", (): void => {
        const position: Position = new Position(1, 2);

        const healerPlayer: HealerPlayer = new HealerPlayer(position);

        const wave: number = 1;
        const requireRepairs: boolean = true;
        const requireLogs: boolean = true;
        const infiniteFood: boolean = false;
        const runnerMovements: Array<string> = [];
        const defenderLevel: number = 5;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            wave,
            requireRepairs,
            requireLogs,
            infiniteFood,
            runnerMovements,
            defenderLevel,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        healerPlayer.tick(barbarianAssault);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const position: Position = new Position(1, 2);

        const healerPlayer: HealerPlayer = new HealerPlayer(position);
        const healerPlayerClone: HealerPlayer = healerPlayer.clone();

        expect(healerPlayerClone).not.toBe(healerPlayer);
        expect(healerPlayerClone.position).not.toBe(healerPlayer.position);
        expect(healerPlayerClone.position.x).toBe(healerPlayer.position.x);
        expect(healerPlayerClone.position.y).toBe(healerPlayer.position.y);
    });

    test("clone is a deep copy with null position", (): void => {
        const position: Position = null;

        const healerPlayer: HealerPlayer = new HealerPlayer(position);
        const healerPlayerClone: HealerPlayer = healerPlayer.clone();

        expect(healerPlayerClone).not.toBe(healerPlayer);
        expect(healerPlayerClone.position).toBe(null);
    });

    test("clone copies code queue and index", (): void => {
        const healerPlayer = new HealerPlayer(new Position(1, 2));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11), new HealerCodeAction(2, 21)];
        healerPlayer.codeIndex = 1;

        const cloned = healerPlayer.clone();

        expect(cloned.codeQueue.length).toBe(2);
        expect(cloned.codeIndex).toBe(1);
        expect(cloned.codeQueue[0]).not.toBe(healerPlayer.codeQueue[0]);
        expect(cloned.codeQueue[0].healerId).toBe(1);
        expect(cloned.codeQueue[1].healerId).toBe(2);
    });
});

describe("processCodeQueue", (): void => {
    function makeBA(): BarbarianAssault {
        return new BarbarianAssault(1, true, true, false, [], 5,
            new Map<number, Array<Command>>(), new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(), new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(), []);
    }

    test("skips action when tick is before waitUntil", (): void => {
        const ba = makeBA();
        ba.ticks = 5;

        const healerPlayer = new HealerPlayer(new Position(30, 30));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(0);
    });

    test("skips dead healer and advances index", (): void => {
        const ba = makeBA();
        ba.ticks = 11;
        // No healers alive — healer 1 doesn't exist
        ba.healers = [];

        const healerPlayer = new HealerPlayer(new Position(30, 30));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(1);
    });

    test("skips dying healer", (): void => {
        const ba = makeBA();
        ba.ticks = 11;
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1, 1);
        healer.isDying = true;
        ba.healers = [healer];

        const healerPlayer = new HealerPlayer(new Position(30, 30));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(1);
    });

    test("uses food when adjacent to healer", (): void => {
        const ba = makeBA();
        ba.ticks = 11;
        const healer = new HealerPenance(new Position(30, 30), 27, 1, 1, 1);
        healer.drawnPosition = new Position(30, 30);
        ba.healers = [healer];

        // Player is south of healer (adjacent)
        const healerPlayer = new HealerPlayer(new Position(30, 29));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(1);
        expect(healer.isPoisoned).toBe(true);
        expect(healer.health).toBe(23); // 27 - 4 from eatFood
    });

    test("does not advance index when not yet done", (): void => {
        const ba = makeBA();
        ba.ticks = 11;
        const healer = new HealerPenance(new Position(30, 30), 27, 1, 1, 1);
        ba.healers = [healer];

        // Player is far from healer — will pathfind but not reach
        const healerPlayer = new HealerPlayer(new Position(20, 20));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(0);
    });
});

// Settings from https://github.com/96jonesa/BaSim/issues/60 (seconds mode): healer 0:37,38 / 46.2:37,37,
// defender 1.2:45,27, healer spawn 42 targeting the defender; food code h1,1 issued at foodTick
function makeStepBackBA(foodTick: number): BarbarianAssault {
    const healerCommands = new Map<number, Array<Command>>();
    healerCommands.set(1, [new MoveCommand(new Position(37, 38))]);
    healerCommands.set(78, [new MoveCommand(new Position(37, 37))]);
    healerCommands.set(foodTick, [new HealerCodeCommand([{healerId: 1, count: 1}])]);
    const defenderCommands = new Map<number, Array<Command>>();
    defenderCommands.set(3, [new MoveCommand(new Position(45, 27))]);

    const ba = new BarbarianAssault(10, true, false, true, [], 5,
        new Map<number, Array<Command>>(), new Map<number, Array<Command>>(),
        healerCommands, new Map<number, Array<Command>>(), defenderCommands, []);
    ba.healerSpawnTargets = ["d"];
    ba.simpleFood = true;
    ba.healerSpawns = [71];

    return ba;
}

describe("arrive delay", (): void => {
    function makeBA(): BarbarianAssault {
        return new BarbarianAssault(10, true, false, true, [], 5,
            new Map<number, Array<Command>>(), new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(), new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(), []);
    }

    test("holds food for a tick after arriving, then uses it without moving", (): void => {
        const ba = makeBA();
        ba.ticks = 80;
        // Healer stepped onto the player's previous tile this tick
        const healer = new HealerPenance(new Position(37, 38), 27, 1, 1, 1);
        healer.drawnPosition = new Position(37, 39);
        ba.healers = [healer];

        // Player arrived at (37, 37) from (37, 38) last tick; stale food path still points at (37, 38)
        const healerPlayer = new HealerPlayer(new Position(37, 37));
        healerPlayer.prevPosition = new Position(37, 38);
        healerPlayer.arrivedLastMove = true;
        healerPlayer.codeQueue = [new HealerCodeAction(1, 0)];
        healerPlayer.findPath(ba, new Position(37, 38));

        healerPlayer.tick(ba);

        expect(healer.isPoisoned).toBe(false);
        expect(healerPlayer.codeIndex).toBe(0);
        expect(healerPlayer.position.equals(new Position(37, 37))).toBe(true);

        // Healer moves on; the player is still adjacent to its drawn tile
        ba.ticks = 81;
        healer.drawnPosition = healer.position.clone();
        healer.position = new Position(38, 38);

        healerPlayer.tick(ba);

        expect(healer.isPoisoned).toBe(true);
        expect(healerPlayer.codeIndex).toBe(1);
        expect(healerPlayer.position.equals(new Position(37, 37))).toBe(true);
    });

    // https://github.com/96jonesa/BaSim/issues/60
    test("player does not move when food lands after a step back", (): void => {
        // Food clicked on the arrival tick (46.8) is held by arrive delay
        const ba = makeStepBackBA(79);

        while (ba.ticks < 82) {
            ba.tick();
            if (ba.ticks >= 79) {
                expect(ba.healerPlayer.position.equals(new Position(37, 37))).toBe(true);
            }
            // Food registers at 48.0 (tick 81), one tick after arrive delay
            expect(ba.healers.length > 0 && ba.healers[0].isPoisoned).toBe(ba.ticks >= 81);
        }
    });

    // Verified in game: waiting out the arrive delay before clicking food still chases the healer
    test("player chases healer when food lands after waiting a tick past the step back", (): void => {
        // Food clicked a tick after arriving (47.4), once the healer is adjacent
        const ba = makeStepBackBA(80);

        while (ba.ticks < 82) {
            ba.tick();
            if (ba.ticks >= 79 && ba.ticks <= 80) {
                expect(ba.healerPlayer.position.equals(new Position(37, 37))).toBe(true);
            }
            // Food still registers at 48.0 (tick 81)
            expect(ba.healers.length > 0 && ba.healers[0].isPoisoned).toBe(ba.ticks >= 81);
        }

        // Player kept following the healer on the tick the food landed
        expect(ba.healerPlayer.position.equals(new Position(38, 37))).toBe(true);
    });
});
