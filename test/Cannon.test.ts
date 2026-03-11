import {describe, expect, test} from "vitest";
import {Cannon, parseCannonInput, getCannonPosition} from "../src/Cannon";
import {CannonCommand} from "../src/CannonCommand";
import {CannonSide} from "../src/CannonSide";
import {PenanceType} from "../src/PenanceType";
import {EggType} from "../src/EggType";
import {BarbarianAssault} from "../src/BarbarianAssault";
import {RunnerPenance} from "../src/RunnerPenance";
import {HealerPenance} from "../src/HealerPenance";
import {RunnerPenanceRng} from "../src/RunnerPenanceRng";
import {Position} from "../src/Position";
import {Command} from "../src/Command";

function makeBA(wave: number = 1): BarbarianAssault {
    return new BarbarianAssault(
        wave, true, true, false, [], 5,
        new Map<number, Array<Command>>(),
        new Map<number, Array<Command>>(),
        new Map<number, Array<Command>>(),
        new Map<number, Array<Command>>(),
        new Map<number, Array<Command>>(),
        []
    );
}

describe("getCannonPosition", (): void => {
    test("west cannon is at (21, 26)", (): void => {
        const pos = getCannonPosition(CannonSide.WEST);
        expect(pos.x).toBe(21);
        expect(pos.y).toBe(26);
    });

    test("east cannon is at (40, 26)", (): void => {
        const pos = getCannonPosition(CannonSide.EAST);
        expect(pos.x).toBe(40);
        expect(pos.y).toBe(26);
    });
});

describe("parseCannonInput", (): void => {
    test("empty string returns empty array", (): void => {
        expect(parseCannonInput("")).toEqual([]);
        expect(parseCannonInput("  ")).toEqual([]);
    });

    test("parses full format wrr,1,51", (): void => {
        const cmds = parseCannonInput("wrr,1,51");
        expect(cmds).not.toBeNull();
        expect(cmds.length).toBe(1);
        expect(cmds[0].cannon).toBe(CannonSide.WEST);
        expect(cmds[0].penance).toBe(PenanceType.RUNNER);
        expect(cmds[0].eggType).toBe(EggType.RED);
        expect(cmds[0].numEggs).toBe(1);
        expect(cmds[0].tick).toBe(51);
    });

    test("parses east cannon healer green egg", (): void => {
        const cmds = parseCannonInput("ehg,3,20");
        expect(cmds).not.toBeNull();
        expect(cmds[0].cannon).toBe(CannonSide.EAST);
        expect(cmds[0].penance).toBe(PenanceType.HEALER);
        expect(cmds[0].eggType).toBe(EggType.GREEN);
        expect(cmds[0].numEggs).toBe(3);
        expect(cmds[0].tick).toBe(20);
    });

    test("parses blue egg", (): void => {
        const cmds = parseCannonInput("wrb,1,10");
        expect(cmds[0].eggType).toBe(EggType.BLUE);
    });

    test("parses multiple commands separated by -", (): void => {
        const cmds = parseCannonInput("wrr,1,11-ehg,2,21");
        expect(cmds).not.toBeNull();
        expect(cmds.length).toBe(2);
        expect(cmds[0].cannon).toBe(CannonSide.WEST);
        expect(cmds[0].eggType).toBe(EggType.RED);
        expect(cmds[0].tick).toBe(11);
        expect(cmds[1].cannon).toBe(CannonSide.EAST);
        expect(cmds[1].eggType).toBe(EggType.GREEN);
        expect(cmds[1].tick).toBe(21);
    });

    test("shorthand inherits from previous command", (): void => {
        const cmds = parseCannonInput("wrr,1,11-2,21");
        expect(cmds).not.toBeNull();
        expect(cmds.length).toBe(2);
        expect(cmds[1].cannon).toBe(CannonSide.WEST);
        expect(cmds[1].penance).toBe(PenanceType.RUNNER);
        expect(cmds[1].eggType).toBe(EggType.RED);
        expect(cmds[1].numEggs).toBe(2);
        expect(cmds[1].tick).toBe(21);
    });

    test("returns null for invalid input", (): void => {
        expect(parseCannonInput("abc")).toBeNull();
        expect(parseCannonInput("wrr,abc,51")).toBeNull();
        expect(parseCannonInput("wrr,1,abc")).toBeNull();
        expect(parseCannonInput("abc,def")).toBeNull();
    });

    test("commands get sequential ids", (): void => {
        const cmds = parseCannonInput("wrr,1,11-2,21-ehg,1,31");
        expect(cmds[0].id).toBe(0);
        expect(cmds[1].id).toBe(1);
        expect(cmds[2].id).toBe(2);
    });
});

describe("Cannon.tick", (): void => {
    test("fires egg at runner in range", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(1);
        expect(runner.eggQueue[0].type).toBe(EggType.RED);
    });

    test("does not fire before command tick", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 10;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(0);
    });

    test("does not fire at dying runner", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        runner.isDying = true;
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(0);
    });

    test("does not fire at stunned runner", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        runner.blueCounter = 5;
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(0);
    });

    test("does not fire at runner out of range", (): void => {
        const ba = makeBA();
        // East cannon at (40,26), place runner far away
        const runner = new RunnerPenance(new Position(10, 10), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(0);
    });

    test("fires at healer when penance type is healer", (): void => {
        const ba = makeBA();
        const healer = new HealerPenance(new Position(30, 26), 27, 1, 1);
        ba.healers.push(healer);
        ba.healersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.HEALER, EggType.GREEN, 1, 11));
        ba.cannon.tick(ba);

        expect(healer.eggQueue.length).toBe(1);
        expect(healer.eggQueue[0].type).toBe(EggType.GREEN);
    });

    test("decrements numEggs after firing", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        const cmd = new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 3, 11);
        ba.cannon.queue.push(cmd);
        ba.cannon.tick(ba);

        expect(cmd.numEggs).toBe(2);
    });

    test("sets stall cooldown after firing", (): void => {
        const ba = makeBA();
        const runner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        const cmd = new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.GREEN, 2, 11);
        ba.cannon.queue.push(cmd);
        ba.cannon.tick(ba);

        expect(cmd.stalled).toBeGreaterThan(0);
    });

    test("targets closer runner over farther runner", (): void => {
        const ba = makeBA();
        // East cannon at (40,26)
        const closerRunner = new RunnerPenance(new Position(38, 26), new RunnerPenanceRng(""), 1, 5);
        const fartherRunner = new RunnerPenance(new Position(30, 26), new RunnerPenanceRng(""), 2, 5);
        ba.runners.push(closerRunner);
        ba.runners.push(fartherRunner);
        ba.runnersAlive = 2;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, EggType.RED, 1, 11));
        ba.cannon.tick(ba);

        expect(closerRunner.eggQueue.length).toBe(1);
        expect(fartherRunner.eggQueue.length).toBe(0);
    });
});

describe("travel time", (): void => {
    test.each([
        // [eggType, distance, expectedTravelTime]
        [EggType.GREEN, 0, 4],
        [EggType.GREEN, 3, 4],
        [EggType.GREEN, 4, 5],
        [EggType.GREEN, 9, 5],
        [EggType.GREEN, 10, 6],
        [EggType.GREEN, 15, 6],
        [EggType.BLUE, 0, 4],
        [EggType.BLUE, 5, 5],
        [EggType.RED, 0, 5],
        [EggType.RED, 3, 5],
        [EggType.RED, 4, 6],
        [EggType.RED, 9, 6],
        [EggType.RED, 10, 7],
        [EggType.RED, 15, 7],
    ])("egg type %s at distance %i has travel time %i", (eggType: EggType, distance: number, expectedTime: number): void => {
        const ba = makeBA();
        // East cannon at (40,26), place runner at exact distance
        const runner = new RunnerPenance(new Position(40 - distance, 26), new RunnerPenanceRng(""), 1, 5);
        ba.runners.push(runner);
        ba.runnersAlive = 1;
        ba.ticks = 11;

        ba.cannon.queue.push(new CannonCommand(0, CannonSide.EAST, PenanceType.RUNNER, eggType, 1, 11));
        ba.cannon.tick(ba);

        expect(runner.eggQueue.length).toBe(1);
        expect(runner.eggQueue[0].stalled).toBe(expectedTime);
    });
});

describe("Cannon.clone", (): void => {
    test("clone is a deep copy", (): void => {
        const cannon = new Cannon();
        cannon.queue.push(new CannonCommand(0, CannonSide.WEST, PenanceType.RUNNER, EggType.RED, 2, 11));

        const cloned = cannon.clone();

        expect(cloned).not.toBe(cannon);
        expect(cloned.queue.length).toBe(1);
        expect(cloned.queue[0]).not.toBe(cannon.queue[0]);
        expect(cloned.queue[0].cannon).toBe(CannonSide.WEST);
        expect(cloned.queue[0].numEggs).toBe(2);

        // Mutating original should not affect clone
        cannon.queue[0].numEggs = 0;
        expect(cloned.queue[0].numEggs).toBe(2);
    });
});
