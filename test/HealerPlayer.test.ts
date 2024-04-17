import {describe, expect, test} from "vitest";
import {HealerPlayer} from "../src/HealerPlayer.js";
import {Position} from "../src/Position.js";
import {BarbarianAssault} from "../src/BarbarianAssault.js";

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
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
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
});