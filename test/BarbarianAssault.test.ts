import {describe, expect, test, vi} from "vitest";
import {BarbarianAssault} from "../src/BarbarianAssault.js";
import {RunnerPenance} from "../src/RunnerPenance.js";
import {Position} from "../src/Position.js";
import {RunnerPenanceRng} from "../src/RunnerPenanceRng.js";
import {FoodType} from "../src/FoodType.js";

describe("constructor", (): void => {
    test.each([
        [1, 2, 2, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [2, 2, 3, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [3, 2, 4, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [4, 3, 4, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [5, 4, 5, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [6, 4, 6, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [7, 5, 6, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [8, 5, 7, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [9, 5, 9, 28, 39, 29, 38, 29, 8, 33, 8, 31, 10, 30, 9, 32, 9],
        [10, 5, 6, 29, 39, 30, 38, 24, 8, 28, 8, 26, 10, 25, 9, 27, 9],
    ])("constructs correctly when wave=%i", (
        wave: number,
        maxRunnersAlive: number,
        totalRunners: number,
        northwestLogsPositionX: number,
        northwestLogsPositionY: number,
        southeastLogsPositionX: number,
        southeastLogsPositionY: number,
        collectorPlayerPositionX: number,
        collectorPlayerPositionY: number,
        defenderPlayerPositionX: number,
        defenderPlayerPositionY: number,
        mainAttackerPlayerPositionX: number,
        mainAttackerPlayerPositionY: number,
        secondAttackerPlayerPositionX: number,
        secondAttackerPlayerPositionY: number,
        healerPlayerPositionX: number,
        healerPlayerPositionY: number,
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            wave,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        expect(barbarianAssault.maxRunnersAlive).toBe(maxRunnersAlive);
        expect(barbarianAssault.totalRunners).toBe(totalRunners);
        expect(barbarianAssault.northwestLogsPosition.x).toBe(northwestLogsPositionX);
        expect(barbarianAssault.northwestLogsPosition.y).toBe(northwestLogsPositionY);
        expect(barbarianAssault.southeastLogsPosition.x).toBe(southeastLogsPositionX);
        expect(barbarianAssault.southeastLogsPosition.y).toBe(southeastLogsPositionY);
        expect(barbarianAssault.collectorPlayer.position.x).toBe(collectorPlayerPositionX);
        expect(barbarianAssault.collectorPlayer.position.y).toBe(collectorPlayerPositionY);
        expect(barbarianAssault.defenderPlayer.position.x).toBe(defenderPlayerPositionX);
        expect(barbarianAssault.defenderPlayer.position.y).toBe(defenderPlayerPositionY);
        expect(barbarianAssault.mainAttackerPlayer.position.x).toBe(mainAttackerPlayerPositionX);
        expect(barbarianAssault.mainAttackerPlayer.position.y).toBe(mainAttackerPlayerPositionY);
        expect(barbarianAssault.secondAttackerPlayer.position.x).toBe(secondAttackerPlayerPositionX);
        expect(barbarianAssault.secondAttackerPlayer.position.y).toBe(secondAttackerPlayerPositionY);
        expect(barbarianAssault.healerPlayer.position.x).toBe(healerPlayerPositionX);
        expect(barbarianAssault.healerPlayer.position.y).toBe(healerPlayerPositionY);
    });
});

describe("tick", (): void => {
    test("tick 1", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.northwestLogsArePresent = false;
        barbarianAssault.southeastLogsArePresent = false;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(1);
        expect(barbarianAssault.northwestLogsArePresent).toBe(false);
        expect(barbarianAssault.southeastLogsArePresent).toBe(false);
        expect(barbarianAssault.currentRunnerId).toBe(1);
        expect(barbarianAssault.runnersAlive).toBe(0);
        expect(barbarianAssault.defenderFoodCall).toBe(FoodType.TOFU);
    });

    test("tick 2", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 1;
        barbarianAssault.northwestLogsArePresent = false;
        barbarianAssault.southeastLogsArePresent = false;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(2);
        expect(barbarianAssault.northwestLogsArePresent).toBe(false);
        expect(barbarianAssault.southeastLogsArePresent).toBe(false);
        expect(barbarianAssault.currentRunnerId).toBe(1);
        expect(barbarianAssault.runnersAlive).toBe(0);
        expect(barbarianAssault.defenderFoodCall).toBe(FoodType.TOFU);
    });

    test("tick 11", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;
        barbarianAssault.northwestLogsArePresent = false;
        barbarianAssault.southeastLogsArePresent = false;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(11);
        expect(barbarianAssault.northwestLogsArePresent).toBe(true);
        expect(barbarianAssault.southeastLogsArePresent).toBe(true);
        expect(barbarianAssault.currentRunnerId).toBe(2);
        expect(barbarianAssault.runnersAlive).toBe(1);
        expect(barbarianAssault.defenderFoodCall).toBe(FoodType.TOFU);
    });

    test("tick 52", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 51;
        barbarianAssault.northwestLogsArePresent = false;
        barbarianAssault.southeastLogsArePresent = false;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(52);
        expect(barbarianAssault.northwestLogsArePresent).toBe(false);
        expect(barbarianAssault.southeastLogsArePresent).toBe(false);
        expect(barbarianAssault.currentRunnerId).toBe(1);
        expect(barbarianAssault.runnersAlive).toBe(0);
        expect(barbarianAssault.defenderFoodCall).not.toBe(FoodType.TOFU);
    });

    test.each([
        [1, 1, 36, 39, 4],
        [1, 2, 36, 39, 5],
        [10, 1, 42, 38, 4],
        [10, 2, 42, 38, 5],
    ])("on wave %i with defender level %i runner spawns at (%i, %i) with sniff distance %i", (
        wave: number,
        defenderLevel: number,
        runnerX: number,
        runnerY: number,
        sniffDistance: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            wave,
            true,
            true,
            false,
            [],
            defenderLevel,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;

        barbarianAssault.tick();

        expect(barbarianAssault.runners.length).toBe(1);

        const runnerPenance: RunnerPenance = barbarianAssault.runners[0];

        expect(runnerPenance.position.x).toBe(runnerX);
        expect(runnerPenance.position.y).toBe(runnerY);
        expect(runnerPenance.sniffDistance).toBe(sniffDistance);
    });

    test("runner doesn't spawn when max are alive", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            2,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;
        barbarianAssault.runners.push(new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 1, 5));
        barbarianAssault.runners.push(new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 2, 5));
        barbarianAssault.runnersAlive = 2;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(11);
        expect(barbarianAssault.runnersAlive).toBe(2);
        expect(barbarianAssault.runners.length).toBe(2);
    });

    test("runner doesn't spawn when all runners already spawned", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            2,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;
        barbarianAssault.runnersKilled = 3;

        barbarianAssault.tick();

        expect(barbarianAssault.ticks).toBe(11);
        expect(barbarianAssault.runnersAlive).toBe(0);
        expect(barbarianAssault.runners.length).toBe(0);
    });

    test("de-spawned runners are removed", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const runner1: RunnerPenance = new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 1, 5);
        const runner2: RunnerPenance = new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 2, 5);
        const runner3: RunnerPenance = new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 3, 5);

        runner1.despawnCountdown = 1;

        barbarianAssault.runners.push(runner1);
        barbarianAssault.runners.push(runner2);
        barbarianAssault.runners.push(runner3);

        barbarianAssault.tick();

        expect(barbarianAssault.runners.length).toBe(2);
        expect(barbarianAssault.runners.indexOf(runner1)).toBe(-1);
        expect(barbarianAssault.runnersToRemove.length).toBe(1);
        expect(barbarianAssault.runnersToRemove[0]).toBe(runner1);
    });

    test("given runner movements are used", (): void => {
        const forcedMovements: Array<string> = ["w"];

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            forcedMovements,
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;
        barbarianAssault.runnerMovementsIndex = 0;

        barbarianAssault.tick();

        expect(barbarianAssault.runners.length).toBe(1);
        expect(barbarianAssault.runners[0].rng.forcedMovements).toBe(forcedMovements[0]);
        expect(barbarianAssault.runnerMovementsIndex).toBe(1);
    });

    test("given runner movements are not used after they have all been used", (): void => {
        const forcedMovements: Array<string> = ["w"];

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            forcedMovements,
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 10;
        barbarianAssault.runnerMovementsIndex = 1;

        barbarianAssault.tick();

        expect(barbarianAssault.runners.length).toBe(1);
        expect(barbarianAssault.runners[0].rng.forcedMovements).toBe("");
    });

    test.each([
        [0.1, FoodType.CRACKERS, FoodType.TOFU],
        [0.6, FoodType.WORMS, FoodType.TOFU],
        [0.1, FoodType.WORMS, FoodType.CRACKERS],
        [0.6, FoodType.TOFU, FoodType.CRACKERS],
        [0.1, FoodType.TOFU, FoodType.WORMS],
        [0.6, FoodType.CRACKERS, FoodType.WORMS],
    ])("random number %d rolls new defender food call %s when current call is %s", (
        randomNumber: number,
        newFoodType: FoodType,
        currentFoodType: FoodType
    ): void => {
        vi.spyOn(Math, "random").mockReturnValue(randomNumber);

        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        barbarianAssault.ticks = 51;
        barbarianAssault.defenderFoodCall = currentFoodType;

        barbarianAssault.tick();

        expect.soft(barbarianAssault.defenderFoodCall).toBe(newFoodType);

        vi.spyOn(Math, "random").mockRestore();
    });
});

describe("tileBlocksPenance", (): void => {
    test("true if position equals defender player position", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const position: Position = barbarianAssault.defenderPlayer.position.clone();

        expect(barbarianAssault.tileBlocksPenance(position)).toBe(true);
    });

    test("true if position equals collector player position", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const position: Position = barbarianAssault.collectorPlayer.position.clone();

        expect(barbarianAssault.tileBlocksPenance(position)).toBe(true);
    });

    test.each([
        [20, 22, 1],
        [21, 22, 1],
        [22, 22, 1],
        [39, 22, 1],
        [40, 22, 1],
        [41, 22, 1],
        [46, 9, 1],
        [46, 10, 1],
        [46, 11, 1],
        [46, 12, 1],
        [27, 24, 1],
        [20, 22, 10],
        [21, 22, 10],
        [22, 22, 10],
        [46, 9, 10],
        [46, 10, 10],
        [46, 11, 10],
        [46, 12, 10],
    ])("true if position is (%i, %i) on wave %i", (
        positionX: number,
        positionY: number,
        wave: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            wave,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const position: Position = new Position(positionX, positionY);

        expect(barbarianAssault.tileBlocksPenance(position)).toBe(true);
    });

    test.each([
        [1, 1, 1],
        [1, 1, 10],
        [39, 22, 10],
        [40, 22, 10],
        [41, 22, 10],
        [27, 24, 10],
    ])("false if position is (%i, %i) on wave %i", (
        positionX: number,
        positionY: number,
        wave: number
    ): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            wave,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const position: Position = new Position(positionX, positionY);

        expect(barbarianAssault.tileBlocksPenance(position)).toBe(false);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const runnerPenance: RunnerPenance = new RunnerPenance(new Position(42, 38), new RunnerPenanceRng(""), 1, 5);

        barbarianAssault.runners.push(runnerPenance);
        barbarianAssault.runnersToRemove.push(runnerPenance);

        const barbarianAssaultClone: BarbarianAssault = barbarianAssault.clone();

        expect(barbarianAssaultClone).not.toBe(barbarianAssault);
        expect(JSON.stringify(barbarianAssaultClone)).toBe(JSON.stringify(barbarianAssault));
    });

    test("clone is a deep copy with null (sub-)fields", (): void => {
        const barbarianAssault: BarbarianAssault = new BarbarianAssault(
            1,
            true,
            true,
            false,
            [],
            5,
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>(),
            new Map<number, Position>()
        );

        const runnerPenance: RunnerPenance = null;

        barbarianAssault.runners.push(runnerPenance);
        barbarianAssault.runnersToRemove.push(runnerPenance);
        barbarianAssault.map = null;
        barbarianAssault.eastTrapPosition = null;
        barbarianAssault.westTrapPosition = null;
        barbarianAssault.defenderPlayer = null;
        barbarianAssault.collectorPlayer = null;
        barbarianAssault.mainAttackerPlayer = null;
        barbarianAssault.secondAttackerPlayer = null;
        barbarianAssault.healerPlayer = null;

        const barbarianAssaultClone: BarbarianAssault = barbarianAssault.clone();

        expect(barbarianAssaultClone).not.toBe(barbarianAssault);
        expect(JSON.stringify(barbarianAssaultClone)).toBe(JSON.stringify(barbarianAssault));
    });
});