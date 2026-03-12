import {describe, expect, test} from "vitest";
import {HealerPlayer} from "../src/HealerPlayer.js";
import {Position} from "../src/Position.js";
import {BarbarianAssault} from "../src/BarbarianAssault.js";
import {Command} from "../src/Command.js";
import {HealerCodeAction} from "../src/HealerCodeAction.js";
import {HealerPenance} from "../src/HealerPenance.js";

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
        const healer = new HealerPenance(new Position(42, 37), 27, 1, 1);
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
        const healer = new HealerPenance(new Position(30, 30), 27, 1, 1);
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
        const healer = new HealerPenance(new Position(30, 30), 27, 1, 1);
        ba.healers = [healer];

        // Player is far from healer — will pathfind but not reach
        const healerPlayer = new HealerPlayer(new Position(20, 20));
        healerPlayer.codeQueue = [new HealerCodeAction(1, 11)];

        healerPlayer.tick(ba);

        expect(healerPlayer.codeIndex).toBe(0);
    });
});