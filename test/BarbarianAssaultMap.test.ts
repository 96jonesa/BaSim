import {describe, expect, test} from "vitest";
import {BarbarianAssaultMap} from "../src/BarbarianAssaultMap.js";
import {Food} from "../src/Food.js";
import {FoodType} from "../src/FoodType.js";
import {Position} from "../src/Position.js";
import {FoodZone} from "../src/FoodZone.js";

describe("reset", (): void => {
    test("removes all food", (): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(1);

        barbarianAssaultMap.addFood(new Food(new Position(1, 2), FoodType.TOFU, true));

        barbarianAssaultMap.reset();

        barbarianAssaultMap.foodZones.forEach((foodZone: FoodZone): void => {
            expect(foodZone.foodList.length).toBe(0);
        });
    });
});

describe("getFoodZone", (): void => {
    test.each([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])("correct food zone when xZone=%i and yZone=%i", (xZone: number, yZone: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(1);

        const foodZone: FoodZone = barbarianAssaultMap.getFoodZone(xZone, yZone);

        expect(foodZone).toBe(barbarianAssaultMap.foodZones[xZone + barbarianAssaultMap.foodZonesWidth * yZone]);
    });
});

describe("addFood", (): void => {
    test.each([
        [0, 0],
        [8, 0],
        [0, 8],
        [8, 8],
    ])("adds food to the correct food zone when position=(%i, %i)", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(1);
        const food: Food = new Food(new Position(x, y), FoodType.TOFU, true);

        barbarianAssaultMap.addFood(food);

        const xZone: number = x >>> 3;
        const yZone: number = y >>> 3;
        const foodZone: FoodZone = barbarianAssaultMap.getFoodZone(xZone, yZone);

        expect(foodZone.foodList.length).toBe(1);
        expect(foodZone.foodList[0]).toBe(food);
    });
});

describe("getFlag", (): void => {
    test.each([
        [29, 6, 0],
        [13, 15, 128],
        [20, 39, 2097184],
    ])("flag at position=(%i, %i) is %i on wave 10", (x: number, y: number, expectedFlag: any): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const flag: number = barbarianAssaultMap.getFlag(new Position(x, y));

        expect(flag).toBe(expectedFlag);
    });
});

describe("canMoveEast", (): void => {
    test.each([
        [27, 38], // can move any direction
        [16, 10], // can only move east
    ])("can move west at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveEast: boolean = barbarianAssaultMap.canMoveEast(new Position(x, y));

        expect(canMoveEast).toBe(true);
    });

    test.each([
        [31, 40], // can only move south or west
        [1, 1], // can not move any direction
    ])("can not move east at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveEast: boolean = barbarianAssaultMap.canMoveEast(new Position(x, y));

        expect(canMoveEast).toBe(false);
    });
});

describe("canMoveWest", (): void => {
    test.each([
        [27, 38], // can move any direction
        [45, 36], // can only move west
    ])("can move west at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveWest: boolean = barbarianAssaultMap.canMoveWest(new Position(x, y));

        expect(canMoveWest).toBe(true);
    });

    test.each([
        [29, 40], // can only move south or east
        [1, 1], // can not move any direction
    ])("can not move west at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveWest: boolean = barbarianAssaultMap.canMoveWest(new Position(x, y));

        expect(canMoveWest).toBe(false);
    });
});

describe("canMoveNorth", (): void => {
    test.each([
        [27, 38], // can move any direction
        [25, 6], // can only move north
    ])("can move west at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveNorth: boolean = barbarianAssaultMap.canMoveNorth(new Position(x, y));

        expect(canMoveNorth).toBe(true);
    });

    test.each([
        [31, 40], // can only move south or west
        [1, 1], // can not move any direction
    ])("can not move east at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveNorth: boolean = barbarianAssaultMap.canMoveNorth(new Position(x, y));

        expect(canMoveNorth).toBe(false);
    });
});

describe("canMoveSouth", (): void => {
    test.each([
        [27, 38], // can move any direction
        [47, 33], // can only move south
    ])("can move west at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveSouth: boolean = barbarianAssaultMap.canMoveSouth(new Position(x, y));

        expect(canMoveSouth).toBe(true);
    });

    test.each([
        [47, 32], // can only move north or east
        [1, 1], // can not move any direction
    ])("can not move east at position=(%i, %i) on wave 10", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const canMoveSouth: boolean = barbarianAssaultMap.canMoveSouth(new Position(x, y));

        expect(canMoveSouth).toBe(false);
    });
});

describe("hasLineOfSight", (): void => {
    test.each([
        [24, 24],
        [35, 39],
        [37, 39],
    ])("has line of sight when un-obstructed and in range (target position=(%i, %i)", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const position1: Position = new Position(36, 39);
        const position2: Position = new Position(x, y);
        const range: number = 15;

        const hasLineOfSight: boolean = barbarianAssaultMap.hasLineOfSight(position1, position2, range);

        expect(hasLineOfSight).toBe(true);
    });

    test("has line of sight when movement-only obstructed and in range", (): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const position1: Position = new Position(36, 39);
        const position2: Position = new Position(47, 26);
        const range: number = 15;

        const hasLineOfSight: boolean = barbarianAssaultMap.hasLineOfSight(position1, position2, range);

        expect(hasLineOfSight).toBe(true);
    });

    test.each([
        [47, 27],
        [30, 42],
        [36, 42],
        [22, 24],
        [22, 27],
    ])("does not have line of sight when vision obstructed and in range (target position=(%i, %i)", (x: number, y: number): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const position1: Position = new Position(36, 39);
        const position2: Position = new Position(x, y);
        const range: number = 15;

        const hasLineOfSight: boolean = barbarianAssaultMap.hasLineOfSight(position1, position2, range);

        expect(hasLineOfSight).toBe(false);
    });

    test("does not have line of sight when out of range", (): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(10);

        const position1: Position = new Position(36, 39);
        const position2: Position = new Position(36, 20);
        const range: number = 15;

        const hasLineOfSight: boolean = barbarianAssaultMap.hasLineOfSight(position1, position2, range);

        expect(hasLineOfSight).toBe(false);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(1);

        barbarianAssaultMap.addFood(new Food(new Position(1, 2), FoodType.TOFU, true));

        const barbarianAssaultMapClone: BarbarianAssaultMap = barbarianAssaultMap.clone();

        expect(barbarianAssaultMapClone).not.toBe(barbarianAssaultMap);
        expect(JSON.stringify(barbarianAssaultMapClone)).toBe(JSON.stringify(barbarianAssaultMap));
    });

    test("clone is a deep copy with null food zone", (): void => {
        const barbarianAssaultMap: BarbarianAssaultMap = new BarbarianAssaultMap(1);

        barbarianAssaultMap.addFood(new Food(new Position(0, 0), FoodType.TOFU, true));

        barbarianAssaultMap.foodZones[1] = null!;

        const barbarianAssaultMapClone: BarbarianAssaultMap = barbarianAssaultMap.clone();

        expect(barbarianAssaultMapClone).not.toBe(barbarianAssaultMap);
        expect(JSON.stringify(barbarianAssaultMapClone)).toBe(JSON.stringify(barbarianAssaultMap));
    });
});