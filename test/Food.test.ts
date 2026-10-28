import {describe, expect, test} from "vitest";
import {Position} from "../src/Position.js";
import {FoodType} from "../src/FoodType.js";
import {Food} from "../src/Food.js";

describe("constructor", (): void => {
    test("green if good", (): void => {
        const position: Position = new Position(1, 2);
        const foodType: FoodType = FoodType.TOFU;
        const isGood: boolean = true;

        const food: Food = new Food(position, foodType, isGood);

        expect(food.colorRed).toBe(0);
        expect(food.colorGreen).toBe(255);
        expect(food.colorBlue).toBe(0);
    });

    test("red if bad", (): void => {
        const position: Position = new Position(1, 2);
        const foodType: FoodType = FoodType.TOFU;
        const isGood: boolean = false;

        const food: Food = new Food(position, foodType, isGood);

        expect(food.colorRed).toBe(255);
        expect(food.colorGreen).toBe(0);
        expect(food.colorBlue).toBe(0);
    });
});

describe("clone", (): void => {
    test("clone is a deep copy", (): void => {
        const position: Position = new Position(1, 2);
        const foodType: FoodType = FoodType.TOFU;
        const isGood: boolean = true;

        const food: Food = new Food(position, foodType, isGood);
        const foodClone: Food = food.clone();

        expect(foodClone).not.toBe(food);
        expect(JSON.stringify(foodClone)).toBe(JSON.stringify(food));
    });

    test("clone is a deep copy with null position", (): void => {
        const position: Position = null!;
        const foodType: FoodType = FoodType.TOFU;
        const isGood: boolean = true;

        const food: Food = new Food(position, foodType, isGood);
        const foodClone: Food = food.clone();

        expect(foodClone).not.toBe(food);
        expect(JSON.stringify(foodClone)).toBe(JSON.stringify(food));
    });
});