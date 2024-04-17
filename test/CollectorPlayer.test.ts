import {describe, expect, test} from "vitest";
import {CollectorPlayer} from "../src/CollectorPlayer.js";
import {Position} from "../src/Position.js";
import {BarbarianAssault} from "../src/BarbarianAssault.js";

describe("tick", (): void => {
    test("does nothing", (): void => {
        const position: Position = new Position(1, 2);

        const collectorPlayer: CollectorPlayer = new CollectorPlayer(position);

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

        collectorPlayer.tick(barbarianAssault);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const position: Position = new Position(1, 2);

        const collectorPlayer: CollectorPlayer = new CollectorPlayer(position);
        const collectorPlayerClone: CollectorPlayer = collectorPlayer.clone();

        expect(collectorPlayerClone).not.toBe(collectorPlayer);
        expect(collectorPlayerClone.position).not.toBe(collectorPlayer.position);
        expect(collectorPlayerClone.position.x).toBe(collectorPlayer.position.x);
        expect(collectorPlayerClone.position.y).toBe(collectorPlayer.position.y);
    });

    test("clone is a deep copy with null position", (): void => {
        const position: Position = null;

        const collectorPlayer: CollectorPlayer = new CollectorPlayer(position);
        const collectorPlayerClone: CollectorPlayer = collectorPlayer.clone();

        expect(collectorPlayerClone).not.toBe(collectorPlayer);
        expect(collectorPlayerClone.position).toBe(null);
    });
});