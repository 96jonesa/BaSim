import {describe, expect, MockInstance, test, vi} from "vitest";
import {RunnerPenance} from "../src/RunnerPenance";
import {Position} from "../src/Position";
import {RunnerPenanceRng} from "../src/RunnerPenanceRng";
import {BarbarianAssault} from "../src/BarbarianAssault";
import {Command} from "../src/Command.js";
import {EggQueueItem} from "../src/EggQueueItem";
import {EggType} from "../src/EggType";
import {CannonSide} from "../src/CannonSide";

describe("tick", (): void => {
    test("cycleTick increases by 1 when less than 10", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.cycleTick = 9;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.cycleTick).toBe(10);
    });

    test("cycleTick resets to 1 after 10", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.cycleTick = 10;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.cycleTick).toBe(1);
    });

    test("ticksStandingStill increases by 1 when position is unchanged", (): void => {

    });

    test("despawnCountdown decreases by 1 when not null", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 9;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.despawnCountdown).toBe(8);
    });

    test("added to runnersToRemove when despawnCountdown reaches 0 (1 at start of tick)", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        runnerPenance.tick(barbarianAssault);

        expect(runnerPenance.despawnCountdown).toBe(0);

        expect(barbarianAssault.runnersToRemove.indexOf(runnerPenance)).not.toBe(-1);
    });

    test("runnersAlive decreases by 1 when not dying and despawnCountdown reaches 0 (1 at start of tick)", (): void => {
        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = false;

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        barbarianAssault.runnersAlive = 1;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.runnersAlive).toBe(0);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("eastTrapCharges decreases by 1 when positive and dying %s east trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        barbarianAssault.eastTrapCharges = 2;

        const position: Position = new Position(barbarianAssault.eastTrapPosition.x + relativeX, barbarianAssault.eastTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.eastTrapCharges).toBe(1);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("eastTrapCharges remains 0 when 0 and dying %s east trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        barbarianAssault.eastTrapCharges = 0;

        const position: Position = new Position(barbarianAssault.eastTrapPosition.x + relativeX, barbarianAssault.eastTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.eastTrapCharges).toBe(0);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("westTrapCharges decreases by 1 when positive and dying %s west trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        barbarianAssault.westTrapCharges = 2;

        const position: Position = new Position(barbarianAssault.westTrapPosition.x + relativeX, barbarianAssault.westTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.westTrapCharges).toBe(1);
    });

    test.each([
        ["on", 0, 0],
        ["north of", 0, 1],
        ["west of", -1, 0],
        ["south of", 0, -1],
        ["east of", 1, 0],
        ["northwest of", -1, 1],
        ["northeast of", 1, 1],
        ["southwest of", -1, -1],
        ["southeast of", 1, -1],
    ])("westTrapCharges remains 0 when 0 and dying %s west trap and despawnCountdown reaches 0 (1 at start of tick)", (
        relativePosition: string,
        relativeX: number,
        relativeY: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            new Map<number, Array<Command>>(),
            []
        );

        barbarianAssault.westTrapCharges = 0;

        const position: Position = new Position(barbarianAssault.westTrapPosition.x + relativeX, barbarianAssault.westTrapPosition.y + relativeY);

        const runnerPenance: RunnerPenance = new RunnerPenance(position, new RunnerPenanceRng(""), 1, 5);

        runnerPenance.despawnCountdown = 1;
        runnerPenance.isDying = true;

        runnerPenance.tick(barbarianAssault);

        expect(barbarianAssault.westTrapCharges).toBe(0);
    });
});

describe("processEggQueue", (): void => {
    test("red egg deals 3 damage when stalled reaches 0", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(2);
    });

    test("red egg stalled counter decrements each call", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.eggQueue.push(new EggQueueItem(3, EggType.RED, CannonSide.WEST));
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(5);
        expect(runner.eggQueue.length).toBe(1);
        expect(runner.eggQueue[0].stalled).toBe(2);
    });

    test("red egg is removed after landing", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        runner.processEggQueue(ba);
        runner.processEggQueue(ba);

        expect(runner.eggQueue.length).toBe(0);
    });

    test("green egg deals 1 damage and starts poison", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.eggQueue.push(new EggQueueItem(0, EggType.GREEN, CannonSide.WEST));
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(4);
        expect(runner.greenCounter).toBe(23);
    });

    test("green egg poison ticks damage every 5 ticks", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.greenCounter = 20;
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(4);
        expect(runner.greenCounter).toBe(19);
    });

    test("green poison does not tick damage at non-multiple of 5", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.greenCounter = 17;
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(5);
        expect(runner.greenCounter).toBe(16);
    });

    test("green counter expires after reaching -1", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.greenCounter = 0;
        runner.processEggQueue(ba);

        expect(runner.greenCounter).toBe(-1);
    });

    test("blue egg sets blueCounter to 9 and clears egg queue", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.eggQueue.push(new EggQueueItem(2, EggType.RED, CannonSide.WEST));
        runner.eggQueue.push(new EggQueueItem(0, EggType.BLUE, CannonSide.EAST));
        runner.processEggQueue(ba);

        expect(runner.blueCounter).toBe(9);
        expect(runner.eggQueue.length).toBe(0);
    });

    test("blue egg on dying runner revives it in place", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);
        runner.isDying = true;

        runner.eggQueue.push(new EggQueueItem(0, EggType.BLUE, CannonSide.EAST));
        runner.processEggQueue(ba);

        expect(runner.isDying).toBe(false);
        expect(runner.position.x).toBe(36);
        expect(runner.position.y).toBe(39);
        expect(runner.despawnCountdown).toBe(null);
        expect(ba.runnersKilled).toBe(0);
    });

    test("blue stun decrements each tick", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.blueCounter = 5;
        runner.tick(ba);

        expect(runner.blueCounter).toBe(4);
    });

    test("runner isDying set when hp reaches 0", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.hp = 3;
        runner.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(0);
        expect(runner.isDying).toBe(true);
    });

    test("runner isDying set when hp goes negative", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.hp = 1;
        runner.eggQueue.push(new EggQueueItem(0, EggType.RED, CannonSide.WEST));
        runner.processEggQueue(ba);

        expect(runner.hp).toBe(-2);
        expect(runner.isDying).toBe(true);
    });

    test("runner skips normal behavior when stunned", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        const ba = new BarbarianAssault(1, true, true, false, [], 5, new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), new Map<number, Array<Command>>(), []);

        runner.blueCounter = 3;
        runner.cycleTick = 5;
        const startX = runner.position.x;
        const startY = runner.position.y;

        runner.tick(ba);

        expect(runner.cycleTick).toBe(5);
        expect(runner.position.x).toBe(startX);
        expect(runner.position.y).toBe(startY);
    });

    test("clone copies egg-related fields", (): void => {
        const runner = new RunnerPenance(new Position(36, 39), new RunnerPenanceRng(""), 1, 5);
        runner.hp = 3;
        runner.blueCounter = 5;
        runner.greenCounter = 10;
        runner.eggQueue.push(new EggQueueItem(3, EggType.RED, CannonSide.WEST));

        const cloned = runner.clone();

        expect(cloned.hp).toBe(3);
        expect(cloned.blueCounter).toBe(5);
        expect(cloned.greenCounter).toBe(10);
        expect(cloned.eggQueue.length).toBe(1);
        expect(cloned.eggQueue[0]).not.toBe(runner.eggQueue[0]);
        expect(cloned.eggQueue[0].stalled).toBe(3);
        expect(cloned.eggQueue[0].type).toBe(EggType.RED);
    });
});