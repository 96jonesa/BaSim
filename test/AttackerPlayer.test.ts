import {describe, expect, test} from "vitest";
import {AttackerPlayer} from "../src/AttackerPlayer.js";
import {Position} from "../src/Position.js";
import {BarbarianAssault} from "../src/BarbarianAssault.js";

describe("tick", (): void => {
    test("does nothing", (): void => {
        const position: Position = new Position(1, 2);

        const attackerPlayer: AttackerPlayer = new AttackerPlayer(position);

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
            new Map<number, Position>(),
            new Map<number, Position>(),
            []
        );

        attackerPlayer.tick(barbarianAssault);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const position: Position = new Position(1, 2);

        const attackerPlayer: AttackerPlayer = new AttackerPlayer(position);
        const attackerPlayerClone: AttackerPlayer = attackerPlayer.clone();

        expect(attackerPlayerClone).not.toBe(attackerPlayer);
        expect(attackerPlayerClone.position).not.toBe(attackerPlayer.position);
        expect(attackerPlayerClone.position.x).toBe(attackerPlayer.position.x);
        expect(attackerPlayerClone.position.y).toBe(attackerPlayer.position.y);
    });

    test("clone is a deep copy with null position", (): void => {
        const position: Position = null;

        const attackerPlayer: AttackerPlayer = new AttackerPlayer(position);
        const attackerPlayerClone: AttackerPlayer = attackerPlayer.clone();

        expect(attackerPlayerClone).not.toBe(attackerPlayer);
        expect(attackerPlayerClone.position).toBe(null);
    });
});